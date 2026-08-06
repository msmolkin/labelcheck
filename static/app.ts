/*
 * labelcheck frontend — one screen, no framework, no external request.
 *
 * Flow: create a batch (optionally with a records CSV) -> upload the images one at a
 * time with three requests in flight -> poll /results with a cursor until the batch is
 * done -> render one row per label, and a review dialog for anything that is not green.
 *
 * Polling rather than server-sent events is a deliberate choice (ARCHITECTURE §7.2): it
 * needs no nginx buffering exceptions and it resumes for free after a page reload,
 * because the cursor lives in the URL rather than in a held connection.
 */

const UPLOAD_CONCURRENCY = 3;
const UPLOAD_RETRIES = 2;
const RETRY_AFTER_MAX_S = 120; // clamp a hostile/broken Retry-After header
const BACKPRESSURE_MAX_MS = 300_000; // give up on a single file after 5 min of 429s
const POLL_ACTIVE_MS = 500;
const POLL_IDLE_MS = 2000;
const IDLE_POLLS_BEFORE_STOP = 3;
const POLL_FAILURES_BEFORE_STOP = 10;
const RESULTS_PAGE = 200;
const BATCH_STORAGE_KEY = "labelcheck.batch";

type Verdict = "green" | "amber" | "red" | "error";

interface Check {
  id: string;
  state: string;
  title: string;
  detail: string;
  citation: string | null;
  evidence: Record<string, unknown>;
  crop: string | null;
}

interface DiffToken {
  w: string;
  s: string;
  expected?: string;
}

interface ImageResult {
  image_id: string;
  filename: string;
  record_id: string | null;
  pairing: string;
  status: string;
  overall: string;
  low_confidence: boolean;
  processing_ms: number;
  passes?: number;
  pass2_reason?: string | null;
  checks: Check[];
  informational: { id: string; text: string }[];
  error?: string | null;
  agent_decision?: string | null;
}

interface ResultsResponse {
  cursor: number;
  done: boolean;
  counts: {
    queued: number;
    running: number;
    green: number;
    amber: number;
    red: number;
    error: number;
  };
  timing: { median_ms: number; p95_ms: number; max_ms: number };
  results: ImageResult[];
}

const VERDICTS: Record<Verdict, { word: string; shape: string; cls: string }> = {
  green: { word: "PASS", shape: "●", cls: "v-green" },
  amber: { word: "NEEDS REVIEW", shape: "▲", cls: "v-amber" },
  red: { word: "FAIL", shape: "■", cls: "v-red" },
  error: { word: "COULD NOT CHECK", shape: "✕", cls: "v-error" },
};

const CHECK_LABELS: Record<string, string> = {
  warning_present: "Warning present",
  warning_text: "Warning text",
  warning_case: "Capitals",
  warning_bold: "Bold",
  brand: "Brand",
  class_type: "Class/type",
  alcohol_content: "Alcohol content",
  net_contents: "Net contents",
};

const DIFF_LEGEND: [string, string][] = [
  ["equal", "read as written"],
  ["noise", "character noise, treated as a match"],
  ["suspect", "one letter off — confirm against the image"],
  ["substituted", "different word on the label"],
  ["missing", "required word not found"],
  ["extra", "extra word inside the statement"],
];

class HttpError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly retryAfterMs: number | null = null,
  ) {
    super(message);
  }
}

/** Seconds from a `Retry-After` header, in ms, clamped so a bad value cannot hang the batch. */
function retryAfterMs(res: Response): number | null {
  const header = res.headers.get("Retry-After");
  if (header === null) return null;
  const seconds = Number.parseFloat(header);
  if (!Number.isFinite(seconds) || seconds < 0) return null;
  return Math.min(seconds, RETRY_AFTER_MAX_S) * 1000;
}

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

/** Server-derived text is always written with textContent, never innerHTML. */
function tag(name: string, className?: string, text?: string): HTMLElement {
  const node = document.createElement(name);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through to the status line */
  }
  return `The server returned ${res.status}.`;
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new HttpError(await readError(res), res.status);
  return (await res.json()) as T;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new HttpError(await readError(res), res.status);
  return (await res.json()) as T;
}

// --- state -----------------------------------------------------------------

let selectedImages: File[] = [];
let recordsFile: File | null = null;
let batchId: string | null = null;
let cursor = 0;
let expectedCount = 0;
let uploadsFinished = true;
let pollTimer = 0;
let idlePolls = 0;
let statuteText: string | null = null;
let activeFilter = "all";
const results = new Map<string, ImageResult>();
const rows = new Map<string, HTMLTableRowElement>();

// --- error banner ----------------------------------------------------------

function showError(message: string): void {
  const banner = el("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clearError(): void {
  el("error-banner").hidden = true;
}

// --- file selection --------------------------------------------------------

function setSelectedImages(files: File[]): void {
  selectedImages = files;
  const summary = el("file-summary");
  summary.textContent =
    files.length === 0
      ? "No images chosen yet."
      : files.length === 1
        ? `1 image chosen: ${files[0].name}`
        : `${files.length} images chosen.`;
  (el("start-button") as HTMLButtonElement).disabled = files.length === 0;
}

function wireFileInputs(): void {
  const input = el<HTMLInputElement>("file-input");
  input.addEventListener("change", () => setSelectedImages(Array.from(input.files ?? [])));

  const records = el<HTMLInputElement>("records-input");
  records.addEventListener("change", () => {
    recordsFile = records.files && records.files.length > 0 ? records.files[0] : null;
    el("records-summary").textContent = recordsFile
      ? `Records file: ${recordsFile.name}`
      : "No records file chosen.";
  });

  // Drag-and-drop is an addition to the file input, never a replacement for it.
  const zone = el("dropzone");
  for (const name of ["dragenter", "dragover"]) {
    zone.addEventListener(name, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    zone.addEventListener(name, () => zone.classList.remove("dragover"));
  }
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    const dropped = Array.from((event as DragEvent).dataTransfer?.files ?? []);
    if (dropped.length > 0) setSelectedImages(dropped);
  });
}

// --- running a batch -------------------------------------------------------

function beginBatch(id: string, expected: number): void {
  batchId = id;
  cursor = 0;
  expectedCount = expected;
  results.clear();
  rows.clear();
  idlePolls = 0;
  el("results-body").replaceChildren();
  el("progress-card").hidden = false;
  el("results-card").hidden = false;
  const exportLink = el<HTMLAnchorElement>("export-link");
  exportLink.href = `/api/batches/${id}/export.csv`;
  exportLink.hidden = false;
  localStorage.setItem(BATCH_STORAGE_KEY, id);
  // The results appear below the fold; move the page to them rather than leaving the
  // user looking at an unchanged screen. No smooth scrolling, so reduced-motion is honoured.
  el("progress-card").scrollIntoView({ block: "start" });
  schedulePoll(0);
}

/** Stops a second batch being started on top of one that is still uploading. */
function setBusy(busy: boolean): void {
  el<HTMLButtonElement>("sample-button").disabled = busy;
  el<HTMLButtonElement>("sample-all-button").disabled = busy;
  el<HTMLButtonElement>("start-button").disabled = busy || selectedImages.length === 0;
}

async function startBatch(): Promise<void> {
  clearError();
  setBusy(true);
  const files = selectedImages;
  const form = new FormData();
  if (recordsFile) form.append("records", recordsFile, recordsFile.name);
  form.append("label", `${files.length} label${files.length === 1 ? "" : "s"}`);
  const res = await fetch("/api/batches", { method: "POST", body: form });
  if (!res.ok) {
    showError(await readError(res));
    setBusy(false);
    return;
  }
  const batch = (await res.json()) as { batch_id: string; warnings?: string[] };
  if (batch.warnings && batch.warnings.length > 0) showError(batch.warnings.join(" "));
  uploadsFinished = false;
  beginBatch(batch.batch_id, files.length);
  try {
    await uploadAll(batch.batch_id, files);
  } finally {
    uploadsFinished = true;
    setBusy(false);
  }
}

async function uploadAll(id: string, files: File[]): Promise<void> {
  let next = 0;
  let accepted = 0;
  const worker = async (): Promise<void> => {
    while (next < files.length) {
      const file = files[next++];
      try {
        const uploaded = await uploadOne(id, file);
        accepted += 1;
        addPendingRow(uploaded.image_id, uploaded.filename || file.name);
      } catch (err) {
        addFailedRow(file.name, err instanceof Error ? err.message : "Upload failed.");
      }
    }
  };
  await Promise.all(Array.from({ length: UPLOAD_CONCURRENCY }, worker));
  expectedCount = accepted;
  try {
    await postJSON(`/api/batches/${id}/close`, { expected_count: accepted });
  } catch {
    /* the batch still completes; the poller reports when it is done */
  }
}

async function uploadOne(id: string, file: File): Promise<{ image_id: string; filename: string }> {
  let attempt = 0;
  let backpressureMs = 0;
  for (;;) {
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const res = await fetch(`/api/batches/${id}/images`, { method: "POST", body: form });
      if (!res.ok) throw new HttpError(await readError(res), res.status, retryAfterMs(res));
      return (await res.json()) as { image_id: string; filename: string };
    } catch (err) {
      // 429 is backpressure, not failure: honour Retry-After and try again without
      // spending a retry — a 300-file batch must be able to wait its turn (§7.5).
      if (err instanceof HttpError && err.status === 429) {
        const wait = err.retryAfterMs ?? 2000;
        backpressureMs += wait;
        if (backpressureMs > BACKPRESSURE_MAX_MS) throw err;
        await sleep(wait);
        continue;
      }
      // Other 4xx means the file itself was rejected (too big, wrong type) — retrying cannot help.
      const permanent = err instanceof HttpError && err.status < 500;
      if (permanent || attempt >= UPLOAD_RETRIES) throw err;
      attempt += 1;
      await sleep(500 * 2 ** attempt);
    }
  }
}

async function loadSampleBatch(size: number): Promise<void> {
  clearError();
  setBusy(true);
  try {
    const batch = await postJSON<{ batch_id: string }>("/api/demo/sample-batch", { size });
    uploadsFinished = true;
    beginBatch(batch.batch_id, size);
  } catch (err) {
    showError(err instanceof Error ? err.message : "Could not load the sample batch.");
  } finally {
    setBusy(false);
  }
}

// --- polling ---------------------------------------------------------------

function schedulePoll(delayMs: number): void {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(poll, delayMs);
}

let pollFailures = 0;

async function poll(): Promise<void> {
  if (!batchId) return;
  let res: ResultsResponse;
  try {
    const url = `/api/batches/${batchId}/results?since=${cursor}&limit=${RESULTS_PAGE}`;
    res = await getJSON<ResultsResponse>(url);
  } catch (err) {
    if (err instanceof HttpError && err.status === 404) {
      // The batch was purged (results are deleted every hour). Start clean.
      localStorage.removeItem(BATCH_STORAGE_KEY);
      batchId = null;
      showError("That batch has been deleted. Uploads are kept for one hour.");
      return;
    }
    pollFailures += 1;
    if (pollFailures >= POLL_FAILURES_BEFORE_STOP) {
      showError("Lost contact with the server. Reload the page to pick the batch back up.");
      return;
    }
    schedulePoll(POLL_IDLE_MS);
    return;
  }
  pollFailures = 0;
  cursor = res.cursor;
  for (const item of res.results) upsertResult(item);
  updateProgress(res);
  // A full page means more rows are already waiting — fetch them before trusting `done`,
  // or a finished 300-image batch would render only its first page after a reload.
  if (res.results.length >= RESULTS_PAGE) {
    schedulePoll(0);
    return;
  }
  if (res.done && uploadsFinished) return;
  const working = res.counts.queued + res.counts.running > 0 || !uploadsFinished;
  idlePolls = working || res.results.length > 0 ? 0 : idlePolls + 1;
  // A batch that was abandoned mid-upload (the tab was closed before it was closed on the
  // server) never reports done. Stop rather than poll an idle batch for ever.
  if (idlePolls >= IDLE_POLLS_BEFORE_STOP) {
    el("progress-eta").textContent = "Checking paused — reload the page to resume.";
    return;
  }
  schedulePoll(working ? POLL_ACTIVE_MS : POLL_IDLE_MS);
}

function updateProgress(res: ResultsResponse): void {
  const c = res.counts;
  const finished = c.green + c.amber + c.red + c.error;
  const total = Math.max(expectedCount, finished + c.queued + c.running);
  const bar = el<HTMLProgressElement>("progress-bar");
  bar.max = Math.max(total, 1);
  bar.value = finished;
  el("progress-text").textContent =
    `${finished} of ${total} checked · ${c.amber} need review · ${c.red} rejected`;
  const remaining = Math.max(total - finished, 0);
  el("progress-eta").textContent =
    res.done && uploadsFinished
      ? "Finished."
      : remaining > 0 && res.timing.median_ms > 0
        ? `About ${seconds(remaining * res.timing.median_ms)} remaining.`
        : "";
  el("timing-footer").textContent = finished
    ? `median ${seconds(res.timing.median_ms)} · slowest ${seconds(res.timing.max_ms)} ·` +
      ` ${finished} checked`
    : "";
}

// --- results table ---------------------------------------------------------

function verdictOf(result: ImageResult): Verdict {
  if (result.status === "error" || result.overall === "error") return "error";
  if (result.overall === "red" || result.overall === "amber" || result.overall === "green") {
    return result.overall;
  }
  return "amber";
}

function verdictCell(verdict: Verdict, low: boolean, agent?: string | null): HTMLElement {
  const cell = tag("td");
  cell.setAttribute("data-label", "Result");
  const style = VERDICTS[verdict];
  const wrap = tag("span", `verdict ${style.cls}`);
  const shape = tag("span", "shape", style.shape);
  shape.setAttribute("aria-hidden", "true");
  wrap.append(shape, tag("span", undefined, style.word));
  cell.append(wrap);
  if (low) cell.append(tag("span", "low-confidence", "checked in reduced mode"));
  if (agent) cell.append(tag("span", "sub", `agent: ${agent}`));
  return cell;
}

function chipsCell(result: ImageResult): HTMLElement {
  const cell = tag("td");
  cell.setAttribute("data-label", "Checks");
  if (result.error) {
    cell.append(tag("span", "sub", result.error));
    return cell;
  }
  const wrap = tag("div", "chips");
  for (const check of result.checks) {
    if (check.state === "na") continue;
    const verdict: Verdict =
      check.state === "green" ? "green" : check.state === "red" ? "red" : "amber";
    const style = VERDICTS[verdict];
    // Shape and (for screen readers) the word carry the state, so the chips stay
    // readable without colour.
    const chip = tag("span", `chip ${style.cls}`);
    const shape = tag("span", undefined, `${style.shape} `);
    shape.setAttribute("aria-hidden", "true");
    chip.append(shape, tag("span", undefined, CHECK_LABELS[check.id] ?? check.id));
    chip.append(tag("span", "visually-hidden", `: ${style.word}`));
    wrap.append(chip);
  }
  cell.append(wrap);
  return cell;
}

function buildRow(imageId: string, filename: string, result?: ImageResult): HTMLTableRowElement {
  const row = document.createElement("tr");

  const thumbCell = tag("td", "cell-thumb");
  const thumb = document.createElement("img");
  thumb.className = "thumb";
  thumb.src = `/api/images/${imageId}/crop/full`;
  thumb.alt = `Label image ${filename}`;
  thumb.loading = "lazy";
  thumbCell.append(thumb);

  const nameCell = tag("td", "filename", filename);
  nameCell.setAttribute("data-label", "File");
  if (result?.record_id) {
    nameCell.append(tag("span", "sub", `record ${result.record_id}`));
  }

  const timeCell = tag("td", "cell-time", result ? seconds(result.processing_ms) : "—");
  timeCell.setAttribute("data-label", "Time");

  const actionCell = tag("td");
  if (result) {
    const button = tag("button", "button button-secondary", "Review") as HTMLButtonElement;
    button.type = "button";
    button.addEventListener("click", () => openReview(imageId));
    actionCell.append(button);
  }

  if (result) {
    row.append(
      thumbCell,
      nameCell,
      verdictCell(verdictOf(result), result.low_confidence, result.agent_decision),
      chipsCell(result),
      timeCell,
      actionCell,
    );
  } else {
    const pending = tag("td", undefined, "Checking…");
    pending.setAttribute("data-label", "Result");
    row.append(thumbCell, nameCell, pending, tag("td"), timeCell, actionCell);
  }
  return row;
}

function placeRow(imageId: string, row: HTMLTableRowElement): void {
  const existing = rows.get(imageId);
  if (existing) existing.replaceWith(row);
  else el("results-body").append(row);
  rows.set(imageId, row);
  applyFilter();
}

function addPendingRow(imageId: string, filename: string): void {
  // The poller may have delivered the finished result between the upload response and
  // this call — a pending row must never overwrite a real one.
  if (results.has(imageId)) return;
  placeRow(imageId, buildRow(imageId, filename));
}

const failedRows: HTMLTableRowElement[] = [];

function addFailedRow(filename: string, message: string): void {
  const row = document.createElement("tr");
  const nameCell = tag("td", "filename", filename);
  nameCell.setAttribute("data-label", "File");
  const detail = tag("td", undefined, message);
  detail.setAttribute("data-label", "Detail");
  row.append(tag("td"), nameCell, verdictCell("error", false), detail, tag("td"), tag("td"));
  el("results-body").append(row);
  failedRows.push(row);
  applyFilter();
}

function upsertResult(result: ImageResult): void {
  results.set(result.image_id, result);
  placeRow(result.image_id, buildRow(result.image_id, result.filename, result));
}

function applyFilter(): void {
  let visible = 0;
  for (const [imageId, row] of rows) {
    const result = results.get(imageId);
    const verdict = result ? verdictOf(result) : "amber";
    const show =
      activeFilter === "all" ||
      (activeFilter === "amber" && (verdict === "amber" || verdict === "error")) ||
      (activeFilter === "red" && verdict === "red");
    row.hidden = !show;
    if (show) visible += 1;
  }
  // Upload failures behave like "needs review": visible on All and Needs review.
  for (const row of failedRows) {
    row.hidden = !(activeFilter === "all" || activeFilter === "amber");
    if (!row.hidden) visible += 1;
  }
  el("results-empty").hidden = visible > 0 || (rows.size === 0 && failedRows.length === 0);
}

function filterButtons(): HTMLButtonElement[] {
  return Array.from(document.querySelectorAll<HTMLButtonElement>(".button-filter"));
}

function wireFilters(): void {
  for (const button of filterButtons()) {
    button.addEventListener("click", () => {
      activeFilter = button.dataset.filter ?? "all";
      for (const other of filterButtons()) {
        other.setAttribute("aria-pressed", String(other === button));
      }
      applyFilter();
    });
  }
}

// --- review dialog ---------------------------------------------------------

/** The check an agent is being asked to adjudicate: the worst one on the image. */
function focusCheck(result: ImageResult): Check | null {
  return (
    result.checks.find((c) => c.state === "red") ??
    result.checks.find((c) => c.state === "amber") ??
    null
  );
}

function renderDiff(diff: DiffToken[]): HTMLElement {
  const wrap = tag("div");
  const legend = tag("p", "legend");
  for (const [state, meaning] of DIFF_LEGEND) {
    legend.append(tag("span", `tok tok-${state}`, `${state}: ${meaning}`));
  }
  const body = tag("p", "diff");
  for (const token of diff) {
    const word = token.s === "missing" && token.expected ? token.expected : token.w;
    const span = tag("span", `tok tok-${token.s}`, `${word} `);
    if (token.expected && token.s !== "missing") {
      span.title = `expected: ${token.expected}`;
    }
    body.append(span);
  }
  wrap.append(legend, body);
  return wrap;
}

function checkBlock(check: Check): HTMLElement {
  const cls = check.state === "green" ? "v-green" : check.state === "red" ? "v-red" : "v-amber";
  const block = tag("div", `check-detail ${cls}`);
  block.append(tag("h3", undefined, check.title));
  if (check.detail) block.append(tag("p", undefined, check.detail));
  block.append(
    tag("p", "cite", check.citation ? check.citation : "Compared against the application record."),
  );
  const diff = check.evidence?.diff as DiffToken[] | undefined;
  if (Array.isArray(diff) && diff.length > 0) block.append(renderDiff(diff));
  return block;
}

function openReview(imageId: string): void {
  const result = results.get(imageId);
  if (!result) return;
  const dialog = el<HTMLDialogElement>("review-dialog");
  const body = el("dialog-body");
  const target = focusCheck(result);

  // An image that could not be checked has nothing to adjudicate.
  const decidable = !result.error;
  el<HTMLButtonElement>("approve-button").disabled = !decidable;
  el<HTMLButtonElement>("reject-button").disabled = !decidable;

  el("review-title").textContent = `Review ${result.filename}`;
  el("dialog-status").textContent = !decidable
    ? (result.error as string)
    : target
      ? `You are reviewing: ${target.title}`
      : "Every check passed. Approving records your agreement.";
  body.replaceChildren();

  const grid = tag("div", "review-grid");
  const crop = document.createElement("img");
  const cropKind = target?.crop ?? "full";
  crop.className = "crop";
  crop.src = `/api/images/${imageId}/crop/${cropKind}`;
  crop.alt = `Region of ${result.filename} that the check was measured on`;
  grid.append(crop);

  const expected = tag("div");
  expected.append(tag("h3", undefined, "Statement required by 27 CFR 16.21"));
  // The statute is prefetched at startup so opening the dialog never waits on the
  // network; on the rare miss the check blocks below still carry the expected words.
  if (statuteText) expected.append(tag("p", "diff", statuteText));
  grid.append(expected);
  body.append(grid);

  // Worst first, so the reason the image was flagged is the first thing on the screen.
  const severity: Record<string, number> = { red: 0, amber: 1, green: 2 };
  const ordered = result.checks
    .filter((check) => check.state !== "na")
    .sort((a, b) => (severity[a.state] ?? 3) - (severity[b.state] ?? 3));
  for (const check of ordered) body.append(checkBlock(check));
  for (const note of result.informational ?? []) {
    body.append(tag("p", "hint", note.text));
  }
  if (result.pass2_reason) {
    body.append(
      tag("p", "hint", `Read a second time at higher magnification: ${result.pass2_reason}.`),
    );
  }
  if (result.low_confidence) {
    body.append(
      tag(
        "p",
        "hint",
        "This image was checked in reduced mode — the slower verification pass was " +
          "skipped to stay inside the time budget, so no rejection is reported.",
      ),
    );
  }

  dialog.dataset.imageId = imageId;
  dialog.dataset.checkId = target ? target.id : "overall";
  dialog.showModal();
  el(decidable ? "approve-button" : "dialog-close").focus();
}

async function adjudicate(decision: "confirm" | "reject"): Promise<void> {
  const dialog = el<HTMLDialogElement>("review-dialog");
  const imageId = dialog.dataset.imageId;
  const checkId = dialog.dataset.checkId;
  if (!imageId || !checkId) return;
  try {
    await postJSON(`/api/images/${imageId}/adjudicate`, { check_id: checkId, decision });
  } catch (err) {
    el("dialog-status").textContent =
      err instanceof Error ? err.message : "Could not record that decision.";
    return;
  }
  // Close first (the native focus restore targets the old Review button while it is
  // still attached), then rebuild the row and move focus onto its replacement button.
  dialog.close();
  const result = results.get(imageId);
  if (result) {
    result.agent_decision = decision === "confirm" ? "approved" : "rejected";
    upsertResult(result);
    rows.get(imageId)?.querySelector("button")?.focus();
  }
}

function wireDialog(): void {
  const dialog = el<HTMLDialogElement>("review-dialog");
  el("approve-button").addEventListener("click", () => void adjudicate("confirm"));
  el("reject-button").addEventListener("click", () => void adjudicate("reject"));
  el("dialog-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("keydown", (event) => {
    const key = event as KeyboardEvent;
    // Bare keys only: Cmd+R (reload) or Ctrl+A (select all) must never record a decision.
    if (key.metaKey || key.ctrlKey || key.altKey) return;
    if (el<HTMLButtonElement>("approve-button").disabled) return;
    const pressed = key.key.toLowerCase();
    if (pressed === "a") void adjudicate("confirm");
    if (pressed === "r") void adjudicate("reject");
  });
}

// --- start-up --------------------------------------------------------------

function resumePreviousBatch(): void {
  const stored = localStorage.getItem(BATCH_STORAGE_KEY);
  if (!stored) return;
  // Re-reading from cursor 0 rebuilds the whole table, so a reload mid-batch can
  // neither duplicate nor lose a row.
  uploadsFinished = true;
  beginBatch(stored, 0);
}

function main(): void {
  wireFileInputs();
  wireFilters();
  wireDialog();
  el("start-button").addEventListener("click", () => void startBatch());
  el("sample-button").addEventListener("click", () => void loadSampleBatch(8));
  el("sample-all-button").addEventListener("click", () => void loadSampleBatch(22));
  // Prefetched so the review dialog opens instantly (never blocked on this request).
  void getJSON<{ text: string }>("/api/statute")
    .then((s) => (statuteText = s.text))
    .catch(() => (statuteText = ""));
  resumePreviousBatch();
}

main();
