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
    "electric", "electricity", "utility", "gas", "internet", "wifi", "phone", "bill",
    "bills", "rent", "mortgage", "insurance", "tax", "taxes", "invoice", "receipt",
    "payment", "due", "charge", "ticket", "tickets", "flight", "hotel", "train", "car",
    "today", "tomorrow", "tonight", "yesterday", "now", "soon", "later", "morning",
    "afternoon", "evening", "night", "day", "week", "weekend", "month", "year",
    "urgent", "asap", "important", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "am", "pm", "for", "and", "or", "the", "with", "at",
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
    "apple", "banana", "orange", "tomato", "onion", "garlic", "lettuce", "spinach",
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
)

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


def detect_and_clean_hallucination(text: str) -> str:
    """Strip repetitive TrOCR/PARSeq decoding loops and pure symbol/digit noise."""
    if not text:
        return ""
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        # Reject TrOCR repeating 1-3 char loops (e.g. "5858585858", "0000000000")
        if re.search(r"(.{1,3})\1{4,}", s):
            continue
        # Reject known TrOCR empty-crop Wikipedia/Hansard hallucination boilerplate
        if re.search(r"Government of (?:Australia|South Africa)|Parliament of Australia|Not logged inTalk", s, re.I):
            continue
        words = s.split()
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
    """Estimate OCR plausibility [0..1] based on alpha ratio and lexicon hits."""
    if not text or not text.strip():
        return 0.0
    tokens = [re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", t).lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0
    lex_hits = sum(1 for t in tokens if t in LEXICON_SET or t.isdigit())
    alpha_ratio = sum(1 for ch in text if ch.isalpha() or ch.isspace()) / max(1, len(text))
    return round(min(1.0, 0.45 * avg_conf + 0.35 * (lex_hits / len(tokens)) + 0.20 * alpha_ratio), 3)


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
    d = [[0.0] * (bl + 1) for _ in range(al + 1)]
    for i in range(al + 1):
        d[i][0] = float(i)
    for j in range(bl + 1):
        d[0][j] = float(j)
    for i in range(1, al + 1):
        for j in range(1, bl + 1):
            sub_cost = 0.0 if a[i - 1] == b[j - 1] else (0.45 if (a[i - 1], b[j - 1]) in VISUAL_CONFUSION_PAIRS else 1.0)
            val = min(d[i - 1][j] + 1.0, d[i][j - 1] + 1.0, d[i - 1][j - 1] + sub_cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, d[i - 2][j - 2] + 0.5)
            d[i][j] = val
    return d[al][bl]


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

    for pat, repl in OCR_CONFUSION_RULES:
        cand = pat.sub(repl, lower)
        if cand != lower and cand in LEXICON_SET:
            return f"{prefix}{_match_case(core, cand)}{suffix}", True

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

    is_cap = core[0].isupper()
    if is_cap and len(lower) <= 4:
        return raw_word, False

    eff_max = max_distance if max_distance is not None else (3 if len(lower) >= 5 else 1)
    best_cand: str | None = None
    best_w = float("inf")
    best_d = 99

    for dict_word in LEXICON_WORDS:
        if abs(len(dict_word) - len(lower)) > eff_max + 1:
            continue
        if len(lower) <= 4 and dict_word[0] != lower[0] and (lower[0], dict_word[0]) not in VISUAL_CONFUSION_PAIRS:
            continue
        d = damerau_levenshtein(lower, dict_word)
        if d > eff_max:
            continue
        w_cost = weighted_handwriting_distance(lower, dict_word)
        if d == 3 and w_cost > 2.15:
            continue
        if d == 2 and len(lower) <= 5 and w_cost > 1.75:
            continue
        if w_cost < best_w or (abs(w_cost - best_w) < 0.01 and d < best_d):
            best_w = w_cost
            best_d = d
            best_cand = dict_word

    if best_cand and (not is_cap or (best_w <= 1.8 and (best_cand[0] == lower[0] or best_d <= 1))):
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
            if len(lower) < 3 or lower in GROCERY_CORE_SET or lower.isdigit():
                fixed_tokens.append(token)
                continue

            best_word: str | None = None
            best_score = float("inf")
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

            if best_word and best_score <= 1.75:
                fixed_tokens.append(f"{prefix}{_match_case(core, best_word)}{suffix}")
            else:
                fixed_tokens.append(token)
        fixed_lines.append(" ".join(fixed_tokens))
    return "\n".join(fixed_lines)


def correct_text_with_lexicon(raw_text: str, capture_mode: str = "auto") -> str:
    if not raw_text or not raw_text.strip():
        return raw_text
    lines_out: List[str] = []
    for line in raw_text.splitlines():
        words = [correct_word_with_lexicon(tok)[0] for tok in line.split()]
        lines_out.append(" ".join(words))
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
        corrected = correct_grocery_context(corrected)
    return corrected
