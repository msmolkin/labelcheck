# labelcheck

A prototype that checks alcohol beverage label images against the federal health warning
requirements (27 CFR Part 16) and against the application record they were submitted
with. An agent drops in one label photo or three hundred, and gets back a traffic light
per label plus the specific reason, usually in about three seconds each.

**Live demo:** https://labelcheck.apps.smolkin.org

Click "Load sample batch" on that page and you'll see results streaming in within a few
seconds. The samples cover the whole range: compliant labels, a title-case "Government
Warning" (rejected), missing bold, a reworded statement, a missing clause, blurry and
glare-damaged photos, and record mismatches.

## What it checks

Four checks come straight from the regulation and need no paperwork:

| Check | Rule |
|---|---|
| Warning present | The health warning statement exists on the label (27 CFR 16.21) |
| Warning text | Word-for-word against the statutory text (16.21) |
| Capitals | "GOVERNMENT WARNING" in capital letters (16.22(a)(2)) |
| Bold | The prefix bold, and the rest of the statement not bold (16.22(a)(2)) |

Four more compare the label against the application record, when you upload one:
brand name, class/type, alcohol content, and net contents.

Every check lands in one of three states. Green means compliant. Red means a real
deviation. Amber means the tool isn't sure and a person should look, and that's a
feature, not an apology: when a photo is too blurred to measure stroke weight, the
honest answer is "you decide," shown next to the cropped evidence.

## Running it locally

You need Python 3.12 and Node 18+ (Node only builds the frontend; nothing runs on it).

```bash
python3 -m venv .venv
.venv/bin/pip install --no-deps --require-hashes -r requirements.txt
npm install && npm run build
.venv/bin/uvicorn app.api:app --port 8712
```

Then open http://127.0.0.1:8712. The `--no-deps` flag matters: requirements.txt is the
complete pinned dependency closure, hashes and all, and installing without it pulls in a
GUI build of OpenCV nobody wants on a server.

The OCR models ship inside the `rapidocr` wheel. After `pip install` the whole thing
runs with networking disabled, which is the point: no cloud APIs, no model downloads at
runtime, nothing leaves the box. There's a test that proves it by breaking `socket`
and running the full pipeline anyway.

## Tests

```bash
make test        # ~2 seconds, no OCR
make test-all    # adds the golden set: 22 bundled labels through the live pipeline
```

The fast suite pins the decision rules (text matching, verdict logic, upload security,
rate limiting). The slow suite runs real images end to end and fails if any label lands
outside its expected verdict, or if the median processing time blows the budget.

## The records CSV

Optional. One row per application:

```
record_id,brand_name,class_type,alcohol_content,net_contents
label_001,OLD TOM DISTILLERY,Straight Bourbon Whiskey,45% Alc./Vol. (90 Proof),750 mL
```

Images pair with records by filename: `label_001.jpg` matches `record_id` `label_001`.
There's a blank template behind the "download template" link on the page. Header
aliases like `brand` or `abv` work too, and the file can come out of Excel with its BOM
and CRLF line endings intact.

## Retention

Uploaded images and results are deleted every hour. This prototype runs on personal
infrastructure, so nothing you upload sticks around, and nothing sensitive should be
uploaded in the first place.

## More detail

The approach, the tools, what was measured, and what this prototype deliberately does
not do: see [docs/APPROACH.md](docs/APPROACH.md). Deployment specifics (systemd unit,
nginx config, the install-time benchmark that tunes resolution to the host) are under
[deploy/](deploy/).
