# Regulatory basis — the health warning statement

Source: 27 CFR Part 16, fetched 2026-08-04 from the official govinfo annual-edition XML
(`CFR-2024-title27-vol1-part16.xml`). This file is the canonical reference for every
string and rule the verifier enforces — do not retype the statement anywhere else;
import it from `app/statute.py` (single source of truth).

## §16.21 — the mandatory statement (byte-exact)

> GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink
> alcoholic beverages during pregnancy because of the risk of birth defects. (2)
> Consumption of alcoholic beverages impairs your ability to drive a car or operate
> machinery, and may cause health problems.

One continuous statement, "separate and apart from all other information," on the brand
label, a separate front label, or a back/side label.

## §16.22 — format requirements

| Rule | Text | Checkable from a photo? |
|---|---|---|
| (a)(1) | Readily legible under ordinary conditions; contrasting background | Partially — legibility/contrast heuristics |
| (a)(2) | "GOVERNMENT WARNING" in CAPITAL letters AND bold type; the **remainder may NOT be bold** | Yes — case from OCR; bold both directions via relative stroke width |
| (a)(3) | Not compressed so as to be illegible | Partially — aspect/character-density heuristic |
| (a)(4) | Max characters per inch: 40 @ 1 mm type, 25 @ 2 mm, 12 @ 3 mm | **No** — needs physical scale (dpi/reference object); documented limitation |
| (b) | Min type size by container: ≤237 mL → 1 mm; >237 mL–3 L → 2 mm; >3 L → 3 mm | **No** — same physical-scale problem; documented limitation |
| (c) | Label firmly affixed | No — not an image property |

Implications for verdicts:

- Text comparison is **exact** against §16.21 after whitespace/line-break normalization
  only. No fuzzy pass is a compliance pass: a perfect match is green; an OCR-plausible
  near-miss is amber (agent confirms against the highlighted crop); anything else is red.
- Case check: the two words "GOVERNMENT WARNING" must be all-caps as read. Title case
  ("Government Warning") is the classic rejection (per the stakeholder notes).
- Bold check is **relative**: mean stroke width (normalized by glyph height) of the
  "GOVERNMENT WARNING" tokens vs the statement body. Prefix not-bolder → violation;
  body as bold as prefix (all-bold) → violation under (a)(2) second sentence.
- Physical-size rules (a)(4)/(b) are honestly out of scope for photo verification and are
  listed as assumptions/limitations in the submission docs, with the container-volume →
  minimum-size table surfaced informationally when net contents is known.
