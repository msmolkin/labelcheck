# Approach, tools, and assumptions

## Reading the requirements

The brief puts its requirements in the interview notes, not the requirements section. So
the first step was extracting them, and every feature in this tool traces to a person in
those notes:

- **Sarah:** results in about 5 seconds, or agents go back to doing it by eye. Batch
  uploads of 200 to 300. A UI her 73-year-old mother could figure out.
- **Marcus:** the network blocks outbound traffic, and the last vendor's cloud ML
  endpoints died on the firewall. So: no cloud APIs. Everything runs on the box.
- **Jenny:** the warning has to be exact, word for word, caps and bold. And it would be
  amazing if the tool could handle photos that aren't perfectly shot.
- **Dave:** STONE'S THROW and Stone's Throw are the same brand. Label review needs
  judgment, and a tool that pretends otherwise makes his life harder.

Two of those pull against each other on purpose. The warning statement is exact because
the law is exact. Brand matching is forgiving because a human would forgive. The tool
holds both by using different comparison rules for different checks, and by giving every
check a third state (amber, "needs review") for the cases where certainty would be
dishonest. Dave's judgment lives in that third state.

## How it works

**Local OCR, two passes.** RapidOCR (PP-OCR models on ONNX Runtime, CPU only, models
shipped inside the wheel) reads every label at 960px on the long side. That first pass
answers most labels in under two seconds. If the recovered statement scores poorly
against the statute, or the read implies a rejection, a second pass crops the statement
band out of the full-resolution image, straightens it using the first pass's detection
geometry, applies contrast equalization, and reads it again. The better read wins. For a
12-megapixel phone photo that crop is a magnification, not a downscale, which is how a
cheap first pass plus a targeted second one beats a single expensive pass on both speed
and accuracy.

The escalation trigger is the statute-match score itself, not the OCR engine's
confidence. I measured the engine's confidence on garbled reads and on perfect reads:
the ranges overlap almost completely (0.987 to 0.995 versus 0.980 to 0.996), so it
carries no signal here. Any read that would reject a label gets the second pass before
the tool is allowed to reject.

**Exact where the law is exact, at the word.** A byte-for-byte comparison sounds right
and is measurably wrong: on my test corpus only 73.5% of images with a perfect warning
produce a perfect OCR read. Byte-exact matching would send one compliant label in four
to the review queue. Instead the tool aligns the read word by word against 27 CFR 16.21
and classifies each difference. Character damage inside a word ("WARNlNG" with a
lowercase L) is noise. A different word ("health issues" for "health problems") is a
violation. A word that's one letter off but not a known OCR confusion goes to a person,
never auto-rejected and never auto-passed. Clause coverage is computed separately, so a
missing clause (2) can't hide behind a good score elsewhere.

**Bold from pixels.** No OCR engine reports font weight, and 16.22(a)(2) requires it
both ways: "GOVERNMENT WARNING" bold, the remainder not bold. The tool measures stroke
width directly. It rectifies each detected word into a flat patch, skeletonizes the ink,
and compares stroke thickness of the prefix words against the statement body. The
thresholds came from experiments on 56 labels across five typefaces: pass above a ratio
of 1.20, fail below 1.05, and a person decides between. Two guards sit in front of the
ratio, because blur physically fattens strokes (at heavy blur a violation can measure
like a pass) and tiny type carries no measurable signal. Both cases go to review with
the reason stated.

**One screen.** Drop zone, a real file input behind a big button, results streaming into
a table as each label finishes, with its processing time on the row. Verdicts are a
shape plus a word plus a color, never color alone. The review dialog shows the cropped
statement next to the statutory text with the differences marked, and two large buttons.
Keyboard works throughout. The page states, above the fold, that uploads are deleted
every hour.

**The agent's decision never overwrites the machine's.** Adjudications are stored
alongside the verdict, and the CSV export carries both columns. A compliance tool that
edits its own findings when a human disagrees has destroyed its audit trail.

## Tools

Python 3.12 and FastAPI for the service. RapidOCR on ONNX Runtime for recognition, held
at two threads to match the deployment host. OpenCV and Pillow for image work. rapidfuzz
for the alignment scores. SQLite (WAL mode) for state; a prototype with one writer does
not need more database than that. The frontend is plain TypeScript compiled with esbuild,
no framework, no external requests of any kind. The Content-Security-Policy is
`default-src 'none'` with only same-origin allowances, so "no phone-home" is verifiable
with one `curl -I`.

## What was measured

Numbers below are from the deployed 2-vCPU host unless noted. The install script
benchmarks the machine before enabling the service and tunes image resolution until the
p95 fits the budget; the health endpoint echoes the chosen settings.

- Per-label processing over the 22-label test set: median 3.8s, p95 4.8s, worst observed
  5.1s. On the development machine: median 2.6s. Sarah's 5-second line holds at the
  median and p95; the single worst image sits at it.
- Warning presence detection: 49 true positives, 7 true negatives, no errors in either
  direction across the corpus. The score margin is wide (100 versus 44 against the
  detection threshold of 80).
- The verdict rules: zero wrong-colored greens or reds across the golden set, on the dev
  machine and on the deployed host. The known-hard cases (heavy blur, tiny type at an
  angle) land in amber by design, and the test suite fails if they ever land anywhere
  else.
- A 300-image batch streams results continuously and finishes in roughly 15 to 20
  minutes. The 5-second requirement is per label, and the first results appear seconds
  after upload starts. Batch throughput on one small box is the honest tradeoff; against
  Sarah's stated 5 to 10 minutes per application done by hand, one box replaces about a
  day of queue time per batch.

## Assumptions and limitations

Stated plainly, because the evaluation criteria ask for them and because the gaps shape
what production work would look like.

1. **Physical type size can't be checked from a photo.** 16.22(b) sets minimum type
   sizes in millimeters and 16.22(a)(4) caps characters per inch. An uncalibrated
   photograph carries no physical scale. The tool shows the required minimums for the
   declared container size as information, and says so. Fixing this needs a reference
   object in frame or a fixed imaging setup.
2. **The test corpus is synthetic.** I generated 56 labels across five font families,
   eight warning conditions, and nine kinds of photo damage, because no labeled corpus
   of real COLA imagery was available on this timeline. Real photos will be harsher:
   curved glass, foil, specular glare, label colors near their background. Thresholds
   should be re-tuned on real imagery before anyone relies on this.
3. **Glare is half-solved.** Contrast equalization recovers washed-out text. Nothing
   recovers pixels that clipped to white. I tried inpainting; it made things worse (it
   deletes the warning along with the glare) and it was cut.
4. **Severe perspective plus small type is the weakest case.** The second-pass crop
   recovers the instances in my corpus, but a real photo at a hard angle with tiny type
   will beat it. Four-point rectification from the label outline is the named next step.
5. **Bold detection is a heuristic, and one edge is open.** A label whose whole
   statement is bold, with an even bolder prefix, could pass the relative test. The
   absolute-weight signal I measured overlaps too much between fonts to gate on without
   flooding the review queue. I chose a heuristic over a trained classifier
   deliberately: there was no labeled training data, and a stroke-width ratio can be
   explained to an agent and an appeals process in one sentence.
6. **No authentication.** It's a prototype on a public URL that stores nothing
   sensitive and purges hourly, per the brief's own framing. It does carry compensating
   controls: strict upload validation and re-encoding, size caps, per-client rate
   limits, a disk quota. Production would sit behind the agency identity provider
   inside the compliance boundary.
7. **No COLA integration, by scope.** The records CSV is shaped like a system export so
   the integration seam is visible.
8. **English only,** matching the models that ship in the wheel.

## What I'd do next

Tune on real label imagery first; everything else is second to that. Then four-point
rectification for angled photos, an active-learning loop that feeds agents' approve and
reject decisions back into the thresholds, and the COLA integration seam turned into a
real importer. The service is one systemd unit and one nginx block, so lifting it into
agency infrastructure is mostly an identity and compliance exercise rather than an
engineering one.
