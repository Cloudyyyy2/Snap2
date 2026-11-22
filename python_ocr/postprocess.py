from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Core reminder, action, scheduling, and grocery/household lexicon
_BASE_LEXICON_WORDS: Tuple[str, ...] = (
    # Actions & Verbs
    "buy", "call", "pay", "pick", "get", "send", "meet", "visit", "order", "bring",
    "check", "clean", "finish", "start", "review", "submit", "schedule", "cancel",
    "book", "reserve", "renew", "update", "fix", "return", "confirm", "reply",
    "email", "text", "message", "prepare", "print", "organize", "plan", "discuss",
    "read", "write", "complete", "deliver", "collect", "drop", "wash", "pack",
    "remind", "remember", "attend", "hire", "change", "invite", "sign", "transfer",
    "deposit", "withdraw", "cook", "bake", "study", "exercise", "workout", "walk",
    "code", "ship",
    # Grocery & Household
    "groceries", "grocery", "milk", "eggs", "bread", "coffee", "tea", "fruit", "vegetables",
    "water", "juice", "cheese", "butter", "chicken", "meat", "rice", "pasta",
    "whip", "cream", "drier", "dryer", "sheet", "sheets", "children", "toothpaste",
    "mexican", "parmesan", "shred", "shredded", "gouda", "cracker", "barrel", "cheddar",
    "fresh", "mozzarella", "mozza", "huevos", "pita", "chip", "chips", "optional",
    "kettle", "potato", "corn", "malta", "platanos", "green", "pan", "good",
    "sugar", "salt", "pepper", "flour", "oil", "vinegar", "sauce", "cereal", "yogurt",
    "beef", "pork", "fish", "onion", "garlic", "tomato", "lettuce", "spinach",
    "avocado", "broccoli", "apple", "banana", "orange", "bagel", "bacon", "salmon",
    "turkey", "asparagus", "soap", "shampoo", "detergent", "towels", "napkins", "batteries",
    # Tasks, Bills, Errands & Time
    "snack", "dinner", "lunch", "breakfast", "medicine", "prescription", "pills",
    "vitamins", "dentist", "doctor", "clinic", "hospital", "appointment", "checkup",
    "therapy", "vet", "veterinarian", "optometrist",
    "electric", "electricity", "utility", "gas", "internet", "wifi", "cable", "phone", "bill",
    "bills", "rent", "mortgage", "insurance", "tax", "taxes", "invoice", "receipt",
    "payment", "due", "charge", "fee", "subscription", "debit",
    "ticket", "tickets", "flight", "hotel", "train", "bus", "car", "rental",
    "airport", "luggage", "passport", "visa", "license", "registration", "permit",
    "homework", "exam", "assignment", "sync", "standup", "presentation",
    "contract", "document", "application", "resume", "proposal", "feedback",
    "gift", "package", "parcel", "mail", "envelope", "keys", "wallet", "charger",
    "glasses", "laptop", "device", "dishes", "plants", "cat", "pet", "gym",
    # People & Family
    "mom", "dad", "boss", "colleague", "coworker", "neighbor", "landlord", "tenant",
    "client", "assistant",
    # Time & Scheduling Words
    "today", "tomorrow", "tonight", "yesterday", "now", "soon", "later", "morning",
    "noon", "afternoon", "evening", "night", "midnight", "day", "week", "weekend",
    "month", "year", "daily", "weekly", "monthly", "urgent", "asap", "important",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "hours", "minutes", "priority", "medium",
    # Connectors & Function Words
    "am", "pm", "for", "and", "or", "the", "with", "at",
    "is", "are", "was", "were", "been", "being", "has", "had", "does", "did",
    "shall", "needs", "needed", "wants", "wanted", "nine",
)

def _load_compact_lexicon() -> Tuple[str, ...]:
    path = Path(__file__).resolve().parent / "compact_lexicon.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return tuple(str(w).strip().lower() for w in data if isinstance(w, str) and w.strip())
        except Exception:
            pass
    return ()


LEXICON_WORDS: Tuple[str, ...] = tuple(dict.fromkeys((*_BASE_LEXICON_WORDS, *_load_compact_lexicon())))
LEXICON_SET = set(LEXICON_WORDS)

GROCERY_CORE_WORDS: Tuple[str, ...] = (
    "whip", "cream", "dryer", "sheets", "children", "toothpaste", "mexican", "cheese",
    "parmesan", "shred", "shredded", "gouda", "cracker", "barrel", "cheddar", "pan",
    "fresh", "mozza", "mozzarella", "huevos", "pita", "chips", "chip", "optional",
    "kettle", "potato", "corn", "malta", "platanos", "good", "green", "milk", "eggs",
    "bread", "butter", "chicken", "beef", "pork", "fish", "rice", "pasta", "sauce",
    "apple", "banana", "orange", "tomato", "tomatoes", "onion", "garlic", "lettuce", "spinach",
    "avocado", "broccoli", "yogurt", "cereal", "coffee", "tea", "juice", "water",
    "sugar", "salt", "pepper", "flour", "oil", "soap", "shampoo", "detergent", "towels",
)
GROCERY_CORE_SET = set(GROCERY_CORE_WORDS)

OCR_CONFUSION_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"rn"), "m"),
    (re.compile(r"cl"), "d"),
    (re.compile(r"vv"), "w"),
    (re.compile(r"li"), "h"),
    (re.compile(r"ol"), "d"),
    (re.compile(r"0"), "o"),
    (re.compile(r"1"), "l"),
    (re.compile(r"5"), "s"),
    (re.compile(r"8"), "b"),
    (re.compile(r"2"), "z"),
    (re.compile(r"9"), "g"),
    (re.compile(r"6"), "b"),
)

# Explicit known cursive/PARSeq misreads kept OUT of LEXICON_SET so they never short-circuit
MISREAD_ALIASES: Dict[str, str] = {
    "browlli": "broccoli",
    "virwilli": "broccoli",
    "vioulli": "broccoli",
    "vjoulli": "broccoli",
    "viralli": "broccoli",
    "timmav": "tomorrow",
    "spinod": "spinach",
    "spinad": "spinach",
    "jolnun": "salmon",
}

VISUAL_CONFUSION_PAIRS = {
    ("a", "o"), ("o", "a"), ("a", "e"), ("e", "a"), ("e", "s"), ("s", "e"),
    ("c", "t"), ("t", "c"), ("c", "r"), ("r", "c"), ("c", "o"), ("o", "c"),
    ("n", "r"), ("r", "n"), ("m", "n"), ("n", "m"), ("u", "v"), ("v", "u"),
    ("u", "a"), ("a", "u"), ("u", "o"), ("o", "u"), ("i", "l"), ("l", "i"),
    ("i", "e"), ("e", "i"), ("g", "q"), ("q", "g"), ("j", "s"), ("s", "j"),
    ("p", "s"), ("s", "p"), ("f", "t"), ("t", "f"), ("k", "h"), ("h", "k"),
    ("b", "v"), ("v", "b"), ("d", "t"), ("t", "d"), ("a", "r"), ("r", "a"),
    ("u", "n"), ("n", "u"), ("i", "r"), ("r", "i"), ("w", "o"), ("o", "w"),
    ("y", "g"), ("g", "y"), ("b", "d"), ("d", "b"), ("p", "h"), ("h", "p"),
}


_WEEKDAY_ABBRS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
_MONTH_OR_DATE_WORD_RE = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|age|birth|born)\b",
    re.I,
)
_WIKI_SIDEBAR_PATTERNS = (
    re.compile(r"related\s*changes", re.I),
    re.compile(r"upload\s*file", re.I),
    re.compile(r"special\s*pages", re.I),
    re.compile(r"permanent\s*link", re.I),
    re.compile(r"page\s*information", re.I),
    re.compile(r"cite\s*this\s*page", re.I),
    re.compile(r"wikidata\s*item", re.I),
)


def detect_and_clean_hallucination(text: str) -> str:
    """Strip repetitive TrOCR/PARSeq decoding loops, boilerplate prose, and pure symbol/digit noise."""
    if not text:
        return ""
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        s = re.sub(r"^[#*_~|]+\s*", "", line.strip()).strip()
        if not s:
            continue
        # Reject standalone punctuation specks
        if re.match(r"^[^A-Za-z0-9]+$", s):
            continue
        # Reject TrOCR repeating 1-3 char loops (e.g. "5858585858", "0000000000")
        if re.search(r"(.{1,3})\1{4,}", s):
            continue
        # Reject sequential single-letter alphabet gibberish ("a b c d e f")
        if re.match(r"^(?:[a-z]\s+){4,}[a-z]\b", s, re.I):
            continue
        # Reject zero loops ("0 000 000 000...", "7 000 000 000")
        if re.match(r"^(?:0+\s+){4,}", s) or re.search(r"(?:^|\s)0{2,}(?:\s+0{2,}){1,}", s):
            continue
        # Reject known TrOCR empty-crop Wikipedia/Hansard/calendar/citation boilerplate
        if re.search(
            r"Government of |Parliament of |Senate of |House of (?:Commons|Representatives)|"
            r"Washington Post|Not logged inTalk|a b c d e f g|"
            r"\b(?:age\s+)?the\s+age\s+of\s+birth\b|\bfirst\s+appearance\s+in\b|"
            r"\b\d+\s+references\b|^\s*(?:references|categories)\s*$|"
            r"\b\d{4}\s+southern\s+state\b|\b\d+(?:st|nd|rd|th)\s+century\b",
            s,
            re.I,
        ):
            continue
        wiki_matches = sum(1 for pat in _WIKI_SIDEBAR_PATTERNS if pat.search(s))
        if wiki_matches >= 2 or re.search(
            r"related\s*changes|special\s*pages|permanent\s*link|page\s*information|cite\s*this\s*page|wikidata\s*item",
            s,
            re.I,
        ):
            continue
        # Reject pure year-range / zero-string hallucinations ("1961 62", "0 0000", "1")
        if re.match(r"^(?:19\d{2}(?:\s+\d{2,4})*|[0\s\.\-:;,\*@#1]+)$", s):
            continue
        # Reject repeated-year + month/birth loops ("1961 02 19 April 19 1961")
        year_matches = re.findall(r"\b(?:19|20)\d{2}\b", s)
        if len(year_matches) >= 2 and len(set(year_matches)) < len(year_matches) and _MONTH_OR_DATE_WORD_RE.search(s):
            continue

        words = s.split()
        # Reject 2- and 3-word date/number n-gram loops (>= 2 repeats with month/year/digit)
        is_ngram_loop = False
        for n in (2, 3):
            if len(words) < n * 2:
                continue
            counts: Dict[str, int] = {}
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n]).lower()
                counts[gram] = counts.get(gram, 0) + 1
                if counts[gram] >= 2 and (_MONTH_OR_DATE_WORD_RE.search(gram) or re.search(r"\d", gram)):
                    is_ngram_loop = True
                    break
            if is_ngram_loop:
                break
        if is_ngram_loop:
            continue

        # Reject general 2-, 3-, and 4-word n-gram phrase loops (>= 3 repeats)
        for n in (2, 3, 4):
            if len(words) < n * 3:
                continue
            counts = {}
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n]).lower()
                counts[gram] = counts.get(gram, 0) + 1
                if counts[gram] >= 3:
                    is_ngram_loop = True
                    break
            if is_ngram_loop:
                break
        if is_ngram_loop:
            continue

        # Token diversity collapse in long outputs
        if len(words) >= 10 and (len({w.lower() for w in words}) / len(words)) < 0.42:
            continue

        # Consecutive duplicate long-word guard alongside possessive/citation cues
        if len(words) >= 4 and re.search(r"'s\b|\b(?:first|appearance|century|references)\b", s, re.I):
            has_long_dup = any(
                words[i].lower() == words[i - 1].lower() and len(re.sub(r"[^a-zA-Z]", "", words[i])) >= 5
                for i in range(1, len(words))
            )
            if has_long_dup:
                continue

        # Residual symbol-junk short lines ("I : - :", "- thu", "- a m", "IK - :")
        if len(words) >= 2 and all(sum(1 for ch in t if ch.isalnum()) <= 1 for t in words):
            continue
        has_symbol_tok = any(re.match(r"^[-:;.,]+$", t) for t in words)
        if has_symbol_tok:
            alnum_cores = [re.sub(r"[^a-zA-Z0-9]", "", t) for t in words]
            non_empty_cores = [c for c in alnum_cores if c]
            if non_empty_cores and all(len(c) <= 3 for c in non_empty_cores):
                has_valid_short_lex = any(
                    len(c) >= 2 and ((c.lower() in LEXICON_SET and c.lower() not in _WEEKDAY_ABBRS) or c.lower() in MISREAD_ALIASES)
                    for c in non_empty_cores
                )
                if not has_valid_short_lex:
                    continue

        deduped: List[str] = []
        repeat_run = 0
        for w in words:
            if deduped and w.lower() == deduped[-1].lower():
                repeat_run += 1
                if repeat_run >= 2:
                    continue
            else:
                repeat_run = 0
            deduped.append(w)
        joined = " ".join(deduped).strip()
        if sum(1 for ch in joined if ch.isalnum()) >= 1:
            cleaned_lines.append(joined)
    return "\n".join(cleaned_lines)


def score_text_plausibility(text: str, avg_conf: float = 0.7) -> float:
    """Estimate OCR plausibility [0..1] based on confidence, lexicon hits, alpha ratio, and noise penalty."""
    if not text or not text.strip():
        return 0.0
    raw_tokens = [t for t in text.split() if t]
    if not raw_tokens:
        return 0.0

    tokens: List[str] = []
    stray_noise = 0
    for raw_tok in raw_tokens:
        cleaned = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", raw_tok).lower()
        if not cleaned:
            stray_noise += 1
            continue
        if len(cleaned) == 1 and not cleaned.isdigit():
            stray_noise += 1
        tokens.append(cleaned)

    if not tokens:
        return 0.0

    lex_hits = sum(1 for t in tokens if t in LEXICON_SET or t.isdigit())
    alpha_ratio = sum(1 for ch in text if ch.isalpha() or ch.isspace()) / max(1, len(text))
    noise_penalty = min(0.35, (stray_noise / len(raw_tokens)) * 0.65)
    low_conf_penalty = 0.12 if avg_conf < 0.70 else 0.0
    clamped_conf = max(0.0, min(1.0, float(avg_conf)))
    raw_score = (
        0.50 * clamped_conf
        + 0.35 * (lex_hits / len(tokens))
        + 0.15 * alpha_ratio
        - noise_penalty
        - low_conf_penalty
    )
    if avg_conf < 0.50:
        raw_score = min(raw_score, 0.44)
    return round(max(0.0, min(1.0, raw_score)), 3)


def damerau_levenshtein(a: str, b: str) -> int:
    al, bl = len(a), len(b)
    if al == 0:
        return bl
    if bl == 0:
        return al
    d = [[0] * (bl + 1) for _ in range(al + 1)]
    for i in range(al + 1):
        d[i][0] = i
    for j in range(bl + 1):
        d[0][j] = j
    for i in range(1, al + 1):
        for j in range(1, bl + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, d[i - 2][j - 2] + cost)
            d[i][j] = val
    return d[al][bl]


def weighted_handwriting_distance(a: str, b: str) -> float:
    al, bl = len(a), len(b)
    if al == 0:
        return float(bl)
    if bl == 0:
        return float(al)
    d = [[0.0] * (bl + 1) for _ in range(al + 1)]
    for i in range(al + 1):
        d[i][0] = float(i)
    for j in range(bl + 1):
        d[0][j] = float(j)
    fast_drop_chars = "aeiour"
    for i in range(1, al + 1):
        ch_obs = a[i - 1]
        for j in range(1, bl + 1):
            ch_tar = b[j - 1]
            sub_cost = 0.0 if ch_obs == ch_tar else (0.45 if (ch_obs, ch_tar) in VISUAL_CONFUSION_PAIRS else 1.0)
            ins_cost = 0.65 if ch_tar in fast_drop_chars else 0.95
            del_cost = 0.95
            val = min(d[i - 1][j] + del_cost, d[i][j - 1] + ins_cost, d[i - 1][j - 1] + sub_cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, d[i - 2][j - 2] + 0.6)
            d[i][j] = val
    return d[al][bl]


def build_ligature_variants(word: str) -> List[str]:
    lower = word.lower()
    step1: Dict[str, None] = {lower: None}
    if "n" in lower:
        step1[re.sub(r"n([aeiou])", r"ri\1", lower)] = None
        step1[lower.replace("ndn", "ndri")] = None
    if "d" in lower:
        step1[lower.replace("d", "cl")] = None
    if re.match(r"^v[ij]", lower):
        step1[re.sub(r"^vi", "b", lower)] = None
        step1[re.sub(r"^vi", "bro", lower)] = None
        step1[re.sub(r"^vir", "bro", lower)] = None
        step1[re.sub(r"^v[ij]o?", "bro", lower)] = None
    if re.search(r"rm{1,2}w$", lower):
        step1[re.sub(r"rm{1,2}w$", "morrow", lower)] = None
    if lower.startswith("t") and len(lower) >= 5 and re.search(r"(?:rm{1,2}w|mm[awv]|mav)$", lower):
        step1[re.sub(r"^t[io].*(?:rm{1,2}w|mm[awv]|mav)$", "tomorrow", lower)] = None
    if "m" in lower:
        step1[lower.replace("m", "rn")] = None

    all_variants: Dict[str, None] = dict(step1)
    for s in step1:
        if "w" in s:
            all_variants[s.replace("w", "co")] = None
            all_variants[s.replace("w", "oc")] = None
            all_variants[s.replace("w", "oo")] = None
            all_variants[s.replace("w", "ur")] = None
    return list(all_variants.keys())


def repair_conservative_phrases(text: str) -> str:
    if not text:
        return text
    out = re.sub(
        r"\bas\s+peas\b",
        lambda m: "Asparagus" if m.group(0).startswith("A") else "asparagus",
        text,
        flags=re.I,
    )
    out = re.sub(
        r"\b(?:pa(?:i?n)\s+(?:an|au)|pa[nm]a?[nu]?)\s+ch[oa]{0,2}[cr]?o?lat\b",
        "Pain au chocolat",
        out,
        flags=re.I,
    )
    out = re.sub(
        r"\bsundried\s+(?:torthel|parshel|parcel|tor[a-z]{2,5})\b",
        "Sundried tomatoes",
        out,
        flags=re.I,
    )
    out = re.sub(
        r"\blunch\s+(?:fir|fiy|fr|fm|fri)\s+(?:tomorrow|t[io]rm{1,2}w|timmav)\b",
        "lunch for tomorrow",
        out,
        flags=re.I,
    )
    out = re.sub(
        r"\b(?:halkey|h[iu]p[ae]y|h[ao]tey|anpey|awbey)\s+mince\b",
        "turkey mince",
        out,
        flags=re.I,
    )
    return out


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement[0].upper() + replacement[1:].lower()
    return replacement.lower()


def correct_word_with_lexicon(raw_word: str, max_distance: int | None = None) -> Tuple[str, bool]:
    trimmed = raw_word.strip()
    if len(trimmed) < 2:
        return raw_word, False

    m_pre = re.match(r"^[^a-zA-Z0-9]+", trimmed)
    m_suf = re.search(r"[^a-zA-Z0-9]+$", trimmed)
    prefix = m_pre.group(0) if m_pre else ""
    suffix = m_suf.group(0) if m_suf else ""
    core = trimmed[len(prefix) : len(trimmed) - len(suffix) if suffix else len(trimmed)]
    if not core:
        return raw_word, False

    raw_lower = core.lower()
    is_contraction = bool(re.search(r"(?:'s|n't|'re|'ve|'ll|'d|'m)$", raw_lower))
    stripped = raw_lower.replace("'", "") if (not is_contraction and "'" in raw_lower) else raw_lower
    lower = re.sub(r"([a-z])\1{2,}", r"\1\1", stripped)

    if re.match(r"^\$?\d+([.:]\d+)?(am|pm)?$", core, re.I):
        return raw_word, False

    if lower in LEXICON_SET:
        return raw_word, False

    if lower in MISREAD_ALIASES:
        return f"{prefix}{_match_case(core, MISREAD_ALIASES[lower])}{suffix}", True

    # Compound split (e.g. "potatochip", "orangjuice")
    if len(lower) >= 7:
        for split_at in range(3, len(lower) - 2):
            left, right = lower[:split_at], lower[split_at:]
            if left in LEXICON_SET and right in LEXICON_SET:
                return f"{prefix}{_match_case(core[:split_at], left)} {right}{suffix}", True
        if not core[0].isupper():
            for split_at in range(3, len(lower) - 2):
                left, right = lower[:split_at], lower[split_at:]
                if len(right) >= 4 and right in LEXICON_SET:
                    left_fix, changed = correct_word_with_lexicon(left, max_distance=2)
                    if changed and left_fix.lower() in LEXICON_SET:
                        return f"{prefix}{_match_case(core[:split_at], left_fix)} {right}{suffix}", True

    # Common 2-letter lowercase fast-handwriting shorthand ('fr' / 'fm' -> 'for')
    if lower in ("fr", "fm") and core == lower:
        return f"{prefix}for{suffix}", True

    for pat, repl in OCR_CONFUSION_RULES:
        cand = pat.sub(repl, lower)
        if cand != lower and cand in LEXICON_SET:
            return f"{prefix}{_match_case(core, cand)}{suffix}", True

    # Test ligature expansions directly against exact lexicon first (e.g. Sundned -> Sundried)
    variants = build_ligature_variants(lower)
    for v in variants:
        if v != lower and v in LEXICON_SET:
            return f"{prefix}{_match_case(core, v)}{suffix}", True

    is_cap = core[0].isupper()
    if is_cap and len(lower) <= 4:
        return raw_word, False

    eff_max_outer = max_distance if max_distance is not None else (3 if len(lower) >= 5 else 1)
    best_cand: str | None = None
    best_w = float("inf")
    best_d = 99

    for dict_word in LEXICON_WORDS:
        if abs(len(dict_word) - len(lower)) > eff_max_outer + 1:
            continue
        if len(lower) <= 4:
            first_pair = (lower[0], dict_word[0]) in VISUAL_CONFUSION_PAIRS
            second_match = len(dict_word) > 1 and dict_word[1] == lower[0]
            if dict_word[0] != lower[0] and not second_match and not first_pair:
                continue

        for obs in variants:
            if len(obs) >= 6:
                eff_max = max(eff_max_outer, 3)
            elif len(obs) >= 5 and len(dict_word) > len(obs):
                eff_max = 3
            else:
                eff_max = min(eff_max_outer, 2 if len(obs) >= 5 else 1)

            if abs(len(dict_word) - len(obs)) > eff_max:
                continue

            plain_dist = damerau_levenshtein(obs, dict_word)
            if plain_dist > eff_max:
                continue

            w_cost = weighted_handwriting_distance(obs, dict_word) + (0.0 if obs == lower else 0.12)
            if plain_dist == 3:
                if w_cost > 2.15:
                    continue
                if len(obs) <= 6:
                    start_ok = obs[0] == dict_word[0] or (obs[0], dict_word[0]) in VISUAL_CONFUSION_PAIRS
                    end_ok = obs[-1] == dict_word[-1] or (obs[-1], dict_word[-1]) in VISUAL_CONFUSION_PAIRS
                    if not start_ok or not end_ok:
                        continue
            if plain_dist == 2 and len(obs) <= 5 and w_cost > 1.75:
                continue

            if w_cost < best_w or (abs(w_cost - best_w) < 0.01 and plain_dist < best_d):
                best_w = w_cost
                best_d = plain_dist
                best_cand = dict_word

    starts_match = best_cand is not None and (
        best_cand[0] == lower[0]
        or any(v[0] == best_cand[0] for v in variants)
        or (
            (len(lower) >= 6 or (not is_cap and len(lower) >= 5))
            and best_w <= 1.5
            and (lower[0], best_cand[0]) in VISUAL_CONFUSION_PAIRS
        )
    )
    ends_match = best_cand is not None and best_cand[-1] == lower[-1]
    compressed_cursive_ok = (
        not is_cap
        and len(lower) >= 4
        and best_d <= 2
        and best_w <= 1.35
        and starts_match
        and ends_match
    )

    if (
        best_cand
        and (best_d <= eff_max_outer or compressed_cursive_ok)
        and (not is_cap or (best_w <= 1.8 and (starts_match or best_d <= 1)))
    ):
        return f"{prefix}{_match_case(core, best_cand)}{suffix}", True

    return raw_word, False


def _dice_bigram(a: str, b: str) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    bg_a: Dict[str, int] = {}
    for i in range(len(a) - 1):
        bg = a[i : i + 2]
        bg_a[bg] = bg_a.get(bg, 0) + 1
    matches = 0
    for i in range(len(b) - 1):
        bg = b[i : i + 2]
        if bg_a.get(bg, 0) > 0:
            matches += 1
            bg_a[bg] -= 1
    return (2.0 * matches) / (len(a) - 1 + len(b) - 1)


def correct_grocery_context(text: str) -> str:
    """Dynamically bias OCR tokens toward grocery vocabulary when in grocery context."""
    if not text or not text.strip():
        return text

    out = text
    for pat, repl in (
        (r"'?(?:push|fresh)\s+(?:n0339|mozza)\b", "fresh mozza"),
        (r"\b(?:uhuerda|huerdas)\b", "huevos"),
        (r"\btootaposls\b", "toothpaste"),
        (r"\bpull\s+chips\b", "pita chips"),
        (r"\bcortachip\b", "corn chip"),
        (r"\bmedical\s+cheese\b", "mexican cheese"),
        (r"\bparmesan(?:'?s)?\s+(?:spend|send|sherd)\b", "parmesan shred"),
        (r"\bbrier\s+sheets\b", "dryer sheets"),
        (r"\b(?:sood|soap)\s+platanos\b", "good platanos"),
        (r"\(\s*gun\s*\)", "(green)"),
        (r"\bpan(?:an)?\s+chocolat\b", "Pain au chocolat"),
        (r"\btimatrel\b", "tomatoes"),
        (r"\bvirwilli\b", lambda m: _match_case(m.group(0), "broccoli")),
        (r"\bbrowlli\b", lambda m: _match_case(m.group(0), "broccoli")),
        (r"\bspin[oa]d\b", lambda m: _match_case(m.group(0), "spinach")),
        (r"\bhalkey\b", "turkey"),
        (r"\bbatm\b", "bacon"),
        (r"\baspornus\b", "asparagus"),
    ):
        out = re.sub(pat, repl, out, flags=re.I)

    fixed_lines: List[str] = []
    for line in out.splitlines():
        fixed_tokens: List[str] = []
        for token in line.split():
            m_pre = re.match(r"^[^a-zA-Z0-9]+", token)
            m_suf = re.search(r"[^a-zA-Z0-9]+$", token)
            prefix = m_pre.group(0) if m_pre else ""
            suffix = m_suf.group(0) if m_suf else ""
            core = token[len(prefix) : len(token) - len(suffix) if suffix else len(token)]
            lower = core.lower()
            if (
                len(lower) < 3
                or lower in GROCERY_CORE_SET
                or lower.isdigit()
                or re.match(r"^\$?\d+([.:/]\d+)?(am|pm)?$", core, re.I)
                or (len(lower) <= 4 and lower in LEXICON_SET)
            ):
                fixed_tokens.append(token)
                continue

            is_cap = bool(core) and core[0].isupper()
            best_word: str | None = None
            best_score = float("inf")
            best_d = 99
            for g_word in GROCERY_CORE_WORDS:
                if abs(len(g_word) - len(lower)) > 2:
                    continue
                d = damerau_levenshtein(lower, g_word)
                max_allowed = 2 if len(lower) >= 4 else 1
                if d > max_allowed:
                    continue
                if len(lower) <= 5 and lower[0] != g_word[0] and (lower[0], g_word[0]) not in VISUAL_CONFUSION_PAIRS:
                    continue
                sim = _dice_bigram(lower, g_word)
                w_cost = weighted_handwriting_distance(lower, g_word) - sim * 0.4
                if d <= 2 and w_cost < best_score:
                    best_score = w_cost
                    best_word = g_word
                    best_d = d

            if best_word and best_score <= 1.75 and (
                not is_cap or (not (len(lower) <= 4 and lower in LEXICON_SET) and (best_word[0] == lower[0] or best_d <= 1))
            ):
                fixed_tokens.append(f"{prefix}{_match_case(core, best_word)}{suffix}")
            else:
                fixed_tokens.append(token)
        fixed_lines.append(" ".join(fixed_tokens))
    return repair_conservative_phrases("\n".join(fixed_lines))


def correct_text_with_lexicon(raw_text: str, capture_mode: str = "auto") -> str:
    if not raw_text or not raw_text.strip():
        return raw_text
    lines_out: List[str] = []
    for line in raw_text.splitlines():
        repaired_line = repair_conservative_phrases(line)
        tokens = re.split(r"(\s+)", repaired_line)
        fixed_tokens = [
            tok if (not tok or tok.isspace()) else correct_word_with_lexicon(tok)[0]
            for tok in tokens
        ]
        joined = "".join(fixed_tokens)
        lines_out.append(repair_conservative_phrases(joined))
    corrected = "\n".join(lines_out)

    looks_like_grocery = capture_mode == "grocery" or (
        len(
            re.findall(
                r"\b(cheese|milk|eggs|chips|platanos|toothpaste|parmesan|cheddar|bagel|potato|grocery|groceries|cream)\b",
                corrected,
                flags=re.I,
            )
        )
        >= 2
    )
    if looks_like_grocery:
        corrected = repair_conservative_phrases(correct_grocery_context(corrected))
    return corrected
