"""Small, conservative French text normalizer for TTS input.

Spells out the things French TTS engines most often trip over in chat
replies: numbers, clock times ("15h30"), ordinals ("1er", "3e"), currency,
percentages and a handful of abbreviations ("M.", "Mme"). Everything else
passes through untouched - this is deliberately not a full normalizer.

Identical copies live in deploy/cpu/kokoro-tts-french/ and
deploy/dgx-spark/kyutai-tts-en_fr/ (each deploy dir is its own Docker build
context). Keep them in sync.
"""

import re

from num2words import num2words

_NBSP = "  "

_ABBREVIATIONS = [
    (re.compile(r"\bM\.(?=\s+[A-ZÀ-Ý])"), "Monsieur"),
    (re.compile(r"\bMM\.(?=\s+[A-ZÀ-Ý])"), "Messieurs"),
    (re.compile(r"\bMmes\b"), "Mesdames"),
    (re.compile(r"\bMme\b"), "Madame"),
    (re.compile(r"\bMlle\b"), "Mademoiselle"),
    (re.compile(r"\bDr\b\.?(?=\s+[A-ZÀ-Ý])"), "Docteur"),
    (re.compile(r"\bPr\b\.?(?=\s+[A-ZÀ-Ý])"), "Professeur"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bp\.\s?ex\."), "par exemple"),
    (re.compile(r"\bc\.-à-d\."), "c'est-à-dire"),
]

# "1 000 000" with (narrow) no-break spaces -> "1000000"; plain spaces are
# left alone because "3 4 5" is just as likely to be a list of numbers.
_THOUSANDS_RE = re.compile(rf"\b\d{{1,3}}(?:[{_NBSP}]\d{{3}})+\b")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3])\s?h(?:\s?([0-5]\d))?\b")
_ORDINAL_RE = re.compile(r"\b(\d+)(er|re|ère|e|ème|eme)\b")
_CURRENCY_RE = re.compile(rf"(\d+)(?:[.,](\d{{1,2}}))?[ {_NBSP}]?(€|\$|£)")
_PERCENT_RE = re.compile(rf"(\d+(?:[.,]\d+)?)[ {_NBSP}]?%")
_DECIMAL_RE = re.compile(r"\b(\d+)[.,](\d+)\b")
_INTEGER_RE = re.compile(r"\b\d+\b")

_CURRENCY_NAMES = {"€": "euros", "$": "dollars", "£": "livres"}


def _cardinal(digits: str) -> str:
    return num2words(int(digits), lang="fr")


def _decimal(match: re.Match) -> str:
    whole, frac = match.group(1), match.group(2)
    # French reads leading zeros in the fractional part digit by digit:
    # "0,05" -> "zéro virgule zéro cinq".
    zeros = len(frac) - len(frac.lstrip("0"))
    frac_words = " ".join(["zéro"] * zeros + ([_cardinal(frac)] if frac.strip("0") else []))
    return f"{_cardinal(whole)} virgule {frac_words or 'zéro'}"


def _number(text: str) -> str:
    text = _DECIMAL_RE.sub(_decimal, text)
    return _INTEGER_RE.sub(lambda m: _cardinal(m.group(0)), text)


def _time(match: re.Match) -> str:
    hours = int(match.group(1))
    minutes = match.group(2)
    unit = "heure" if hours <= 1 else "heures"
    # "heure" is feminine: 1h -> "une heure", 21h -> "vingt et une heures".
    spoken = re.sub(r"\bun$", "une", _cardinal(str(hours))) + f" {unit}"
    if minutes and int(minutes) > 0:
        spoken += f" {_cardinal(minutes)}"
    return spoken


def _currency(match: re.Match) -> str:
    # "12,50 €" -> "douze euros cinquante", the way it is said aloud.
    whole, cents, symbol = match.group(1), match.group(2), match.group(3)
    spoken = f"{_cardinal(whole)} {_CURRENCY_NAMES[symbol]}"
    if cents and int(cents) > 0:
        spoken += f" {_cardinal(cents.ljust(2, '0'))}"
    return spoken


def _ordinal(match: re.Match) -> str:
    n, suffix = int(match.group(1)), match.group(2)
    if n == 1:
        return "première" if suffix in ("re", "ère") else "premier"
    return num2words(n, lang="fr", to="ordinal")


def normalize_french(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    text = _THOUSANDS_RE.sub(lambda m: re.sub(rf"[{_NBSP}]", "", m.group(0)), text)
    text = _TIME_RE.sub(_time, text)
    text = _ORDINAL_RE.sub(_ordinal, text)
    text = _CURRENCY_RE.sub(_currency, text)
    text = _PERCENT_RE.sub(lambda m: f"{_number(m.group(1))} pour cent", text)
    return _number(text)
