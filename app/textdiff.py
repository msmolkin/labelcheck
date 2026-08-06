"""Word-level comparison of the recovered statement against 27 CFR 16.21.

A byte-exact comparison is the obvious rule and it is measurably wrong: only 73.5 % of
warning-bearing corpus images produce an exact token match (§2.6), so it would deny green to
about one compliant label in four. This keeps the word-for-word requirement at the *word*, the
legal unit, and tolerates character damage inside a word, where OCR error lives.

Three independent signals come out: the classified **diff**, the per-clause **coverage**
(matched directly rather than through the alignment, so a poor alignment cannot hide a missing
clause (2)), and how much **noise** the read carried. None of them decides the verdict alone.
"""

from __future__ import annotations

import difflib
import string
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from app.config import Settings
from app.statute import CLAUSE_1, CLAUSE_2, FUNCTION_WORDS, STATUTE_TOKENS

# Characters OCR interchanges, folded onto one representative each (§6.2). Applied after
# lowercasing, so "S" and "s" have already met. `rn`->`m`, `cl`->`d` and `vv`->`w` are
# two-character confusions and run first, before the single characters move.
_PAIR_FOLD = (("rn", "m"), ("cl", "d"), ("vv", "w"))
_CHAR_FOLD = str.maketrans({"o": "0", "l": "1", "i": "1", "s": "5", "b": "8"})
_CONFUSABLE = set("0o1liI5S8BrnmcdvwO")
_PUNCT_FOLD = str.maketrans({**dict.fromkeys("‘’‛′´`", "'"), **dict.fromkeys("‐‑‒–—―−", "-")})
_STRIP = string.punctuation + "“”"

# §6.2: clause coverage is a rule of its own, not a tuning knob — 0.9 is what green requires
# and anything under 0.5 means the clause is not on the label at all.
CLAUSE_COVERAGE_GREEN = 0.9
CLAUSE_COVERAGE_RED = 0.5

_CLAUSE_TOKENS = {"1": CLAUSE_1.split(), "2": CLAUSE_2.split()}


def fold(token: str) -> str:
    """Reduce a token to the form in which OCR damage no longer shows: case, accents,
    surrounding punctuation and the confusion classes all collapse, while two genuinely
    different words stay apart ("problems" -> `pr08lem5` against "issues" -> `155ue5`)."""
    plain = unicodedata.normalize("NFKD", token or "")
    plain = "".join(c for c in plain if not unicodedata.combining(c))
    plain = plain.translate(_PUNCT_FOLD).strip(_STRIP).lower()
    for pair, single in _PAIR_FOLD:
        plain = plain.replace(pair, single)
    return plain.translate(_CHAR_FOLD)


def is_noise(expected: str, got: str, s: Settings, *, edge: bool = False) -> bool:
    """True when the difference between two tokens is OCR damage rather than a rewording.

    Two ways to qualify: the folded forms are equal, or the pair is similar enough
    (`noise_word_min_sim`) *and* every differing character belongs to a confusion class. `edge`
    is a third: the statement is located by a fuzzy character-span alignment, so its first and
    last tokens can be cut in half by that boundary, which is an artefact of where we cut
    rather than something the label got wrong.
    """
    a, b = fold(expected), fold(got)
    if a == b:
        return True
    if edge and b and (a.startswith(b) or a.endswith(b)):
        return True
    if fuzz.ratio(a, b) < s.noise_word_min_sim:
        return False
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return all(
        tag == "equal" or set(a[i1:i2] + b[j1:j2]) <= _CONFUSABLE for tag, i1, i2, j1, j2 in ops
    )


def _bare(token: str) -> str:
    """The token as a plain lowercase word, for the function-word test."""
    return token.translate(_PUNCT_FOLD).strip(_STRIP).lower()


@dataclass(frozen=True)
class Defect:
    """One real deviation from the statutory wording."""

    kind: str  # "substituted" | "missing" | "extra"
    expected: str
    got: str
    critical: bool  # False only for the seven function words (§6.2)


@dataclass
class TextDiff:
    """The full comparison: a renderable diff plus the counts the verdict rule reads."""

    state: str  # green | amber | red, assuming the read is trusted
    diff: list[dict] = field(default_factory=list)
    defects: list[Defect] = field(default_factory=list)
    clause_coverage: dict[str, float] = field(default_factory=dict)
    noise_words: int = 0
    noise_ratio: float = 0.0
    suspects: list[tuple[str, str]] = field(default_factory=list)  # (expected, got)

    @property
    def suspect_words(self) -> int:
        return len(self.suspects)

    @property
    def critical_defects(self) -> int:
        return sum(1 for d in self.defects if d.critical)

    @property
    def function_defects(self) -> int:
        return sum(1 for d in self.defects if not d.critical)

    @property
    def function_missing(self) -> int:
        """Function words absent from the read — OCR drops small words constantly."""
        return sum(1 for d in self.defects if not d.critical and d.kind == "missing")

    @property
    def function_altered(self) -> int:
        """Function words swapped or inserted — a different animal from a dropped one:
        "the ability" printed for "your ability" is a wording change, not read damage."""
        return sum(1 for d in self.defects if not d.critical and d.kind != "missing")

    @property
    def missing_clauses(self) -> list[str]:
        """Clauses whose words are simply not on the label (§6.2)."""
        return [n for n, c in sorted(self.clause_coverage.items()) if c < CLAUSE_COVERAGE_RED]


def clause_coverage(read_tokens: list[str], s: Settings) -> dict[str, float]:
    """Fraction of each clause's words present in the read, matched independently of the
    alignment: exact folded matches are consumed first, then the plausible ones (§6.2)."""
    coverage: dict[str, float] = {}
    for number, clause in _CLAUSE_TOKENS.items():
        pool = [fold(t) for t in read_tokens]
        wanted = [fold(t) for t in clause]
        hits = 0
        for word in list(wanted):
            if word in pool:
                pool.remove(word)
                wanted.remove(word)
                hits += 1
        for word in wanted:
            match = next((p for p in pool if is_noise(word, p, s)), None)
            if match is not None:
                pool.remove(match)
                hits += 1
        coverage[number] = hits / len(clause)
    return coverage


def compare(region_text: str, s: Settings) -> TextDiff:
    """Align the read against 27 CFR 16.21 and classify every difference.

    `autojunk=False` is explicit: it only fires on 200+ element sequences and the statute is
    43 tokens, but a future reader should not have to work that out.
    """
    read = (region_text or "").split()
    statute = list(STATUTE_TOKENS)
    last = len(statute) - 1
    result = TextDiff(state="amber")

    def _record(kind: str, expected: str, got: str) -> None:
        result.defects.append(
            Defect(kind, expected, got, _bare(expected or got) not in FUNCTION_WORDS)
        )

    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, statute, read, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            result.diff += [{"w": read[j], "s": "equal"} for j in range(j1, j2)]
            continue
        if tag == "insert":
            # Extra words before the first or after the last statutory word are the label
            # copy either side of the statement bleeding into the crop, not an insertion.
            if i1 not in (0, len(statute)):
                result.diff += [{"w": read[j], "s": "extra"} for j in range(j1, j2)]
                for j in range(j1, j2):
                    _record("extra", "", read[j])
            continue
        paired = min(i2 - i1, j2 - j1)  # "replace" blocks line up word for word as far as they go
        for k in range(paired):
            expected, got = statute[i1 + k], read[j1 + k]
            if is_noise(expected, got, s, edge=i1 + k in (0, last)):
                result.noise_words += 1
                result.diff.append({"w": got, "s": "noise", "expected": expected})
            elif fuzz.ratio(fold(expected), fold(got)) >= s.noise_word_min_sim:
                # Nearly the required word, but the differing glyphs are outside the
                # confusion classes — "probloms" for "problems." (caught on the deploy
                # box at 832 px). Neither a match nor proof of a rewording: a suspect
                # forces amber, never red and never green (§6.2).
                result.suspects.append((expected, got))
                result.diff.append({"w": got, "s": "suspect", "expected": expected})
            else:
                result.diff.append({"w": got, "s": "substituted", "expected": expected})
                _record("substituted", expected, got)
        for expected in statute[i1 + paired : i2]:
            result.diff.append({"w": expected, "s": "missing"})
            _record("missing", expected, "")
        for got in read[j1 + paired : j2]:
            result.diff.append({"w": got, "s": "extra"})
            _record("extra", "", got)

    result.noise_ratio = result.noise_words / len(statute)
    result.clause_coverage = clause_coverage(read, s)
    result.state = _state(result, s)
    return result


def _state(d: TextDiff, s: Settings) -> str:
    """§6.2's rule. Red still has to pass the TRUST gate in `app.verdict` to be reported."""
    worst_clause = min(d.clause_coverage.values(), default=0.0)
    if (
        d.critical_defects == 0
        and d.suspect_words == 0
        and d.function_missing <= 1
        and d.function_altered == 0
        and d.noise_ratio <= s.noise_green_max_ratio
        and worst_clause >= CLAUSE_COVERAGE_GREEN
    ):
        return "green"
    if d.critical_defects >= 1 or worst_clause < CLAUSE_COVERAGE_RED:
        return "red"
    return "amber"
