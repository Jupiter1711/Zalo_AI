"""Chuẩn hóa số tiếng Việt cho pipeline ASR -> LLM (chỉ thư viện chuẩn).

- verbalize(text): chữ số -> chữ ("50 nghìn" -> "năm mươi nghìn", "0983 214 6755" -> đọc từng số). Mã chữ+số
  (VTV6, RON95, RV219, 24h, 8A3...) giữ nguyên. Dùng để làm target ASR thống nhất, vì transcript gốc ghi lẫn số/chữ.
- normalize_asr(text): chữ thường, bỏ dấu câu -> dạng chữ mà ASR sẽ xuất ra.
- itn(text): chữ -> chữ số, chỉ ở chỗ chắc chắn là số:
    * cụm có từ cấu trúc (mười, mươi, chục, trăm, nghìn/ngàn, triệu, tỷ): "hai triệu tám" -> "2800000"
    * chuỗi >= 4 chữ số đọc liền (SĐT, vé số): "không chín năm ..." -> "0950..."
    * một chữ số đứng trước đơn vị (giờ, độ, tầng, người...) hoặc sau "tầng/số/mùng/lầu/kênh": "ba mét" -> "3 mét"
  Các từ đa nghĩa đứng một mình (năm nay, thứ tư, ngày mốt, ba mẹ, một chút) giữ nguyên.
- asr_target(text) = normalize_asr(verbalize(text)); llm_input(asr_text) = itn(normalize_asr(asr_text)).
"""
from __future__ import annotations

import re
import unicodedata

DIGIT_WORDS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

# ---------------------------------------------------------------- verbalize: số -> chữ

SCALES = [(10**9, "tỷ"), (10**6, "triệu"), (10**3, "nghìn")]


def _read_under_1000(n: int, full: bool) -> list[str]:
    """0 < n < 1000. full=True: đọc đủ "không trăm"/"linh" khi đứng sau một nhóm lớn hơn (1050 -> ... không trăm năm mươi)."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    out = []
    if h or full:
        out += [DIGIT_WORDS[h], "trăm"]
    if t == 0:
        if u and out:
            out.append("linh")
    elif t == 1:
        out.append("mười")
    else:
        out += [DIGIT_WORDS[t], "mươi"]
    if u:
        if u == 1 and t >= 2:
            out.append("mốt")
        elif u == 4 and t >= 2:
            out.append("tư")
        elif u == 5 and t >= 1:
            out.append("lăm")
        else:
            out.append(DIGIT_WORDS[u])
    return out


def read_number(n: int) -> str:
    """Số nguyên không âm -> chữ: 26 -> "hai mươi sáu", 2026 -> "hai nghìn không trăm hai mươi sáu"."""
    if n == 0:
        return "không"
    words, started = [], False
    for scale, name in SCALES:
        if n >= scale:
            words += [*(_read_under_1000(n // scale, started) if n // scale < 1000 else read_number(n // scale).split()), name]
            n %= scale
            started = True
    if n:
        words += _read_under_1000(n, started)
    return " ".join(words)


def read_digits(s: str) -> str:
    return " ".join(DIGIT_WORDS[int(c)] for c in s if c.isdigit())


# Chuỗi nhóm chữ số cách nhau bởi khoảng trắng (SĐT "0983 214 6755") hoặc một nhóm số, không dính chữ cái hai bên.
_NUM_RUN = re.compile(r"(?<![\w-])\d+(?:[  ]\d+)*(?!\w)")


def _verbalize_run(m: re.Match) -> str:
    run = m.group()
    groups = run.split()
    digits = "".join(groups)
    if len(groups) > 1 or (len(digits) > 1 and digits[0] == "0") or len(digits) >= 5:
        return read_digits(digits)  # SĐT, số vé, mã: đọc từng chữ số
    return read_number(int(digits))


def verbalize(text: str) -> str:
    return _NUM_RUN.sub(_verbalize_run, unicodedata.normalize("NFC", text))


# ---------------------------------------------------------------- normalize_asr

_PUNCT = re.compile(r"[.,?!:;…\"“”()]")
_SPACE = re.compile(r"\s+")


def normalize_asr(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("-", " ")
    return _SPACE.sub(" ", _PUNCT.sub(" ", text)).strip()


def asr_target(text: str) -> str:
    return normalize_asr(verbalize(text))


# ---------------------------------------------------------------- itn: chữ -> số

DIGIT = {w: i for i, w in enumerate(DIGIT_WORDS)} | {"bẩy": 7}
UNIT_FORMS = {"mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5}  # chỉ hợp lệ sau mươi/mười
SCALE = {"nghìn": 10**3, "ngàn": 10**3, "triệu": 10**6, "tỷ": 10**9, "tỉ": 10**9}
STRUCT = {"mười", "mươi", "chục", "trăm", "linh", "lẻ", "rưỡi"} | set(SCALE)
NUMBER_WORDS = set(DIGIT) | set(UNIT_FORMS) | STRUCT
UNITS = {
    "giờ", "phút", "giây", "tiếng", "độ", "phần", "ngày", "tuần", "tháng", "người", "chỗ", "vé", "tầng", "lầu",
    "mét", "km", "cây", "lít", "ký", "kg", "cân", "lần", "đô", "đồng", "bước", "buổi", "kỳ", "suất", "xuất",
    "phòng", "bài", "hộp", "chai", "ly", "cuộn", "xấp", "gói", "hũ", "quả", "trái", "kí", "tấn", "điểm", "%",
    "rưỡi",  # "năm rưỡi" -> "5 rưỡi" (5 giờ 30)
}
PREFIX = {"tầng", "lầu", "số", "mùng", "kênh"}


class _Fail(Exception):
    pass


def _parse_cardinal(tokens: list[str]) -> int:
    """Cụm số đọc theo cấu trúc -> giá trị. Raise _Fail nếu không đọc được rõ ràng."""
    total, cur, last, last_scale = 0, 0, None, None
    i = 0
    while i < len(tokens):
        w = tokens[i]
        if w in DIGIT or w in UNIT_FORMS:
            d = DIGIT.get(w, UNIT_FORMS.get(w))
            if w in UNIT_FORMS and last != "tens":
                raise _Fail
            if last == "digit":
                raise _Fail  # "hai chín": hai chữ số liền nhau, không rõ nghĩa
            if last == "trăm":  # "trăm hai" = 120, "hai trăm năm mươi" = 250
                cur += d * 10
                last = "tens_digit"
            elif last == "scale" and (i + 1 == len(tokens)):  # "hai triệu tám" = 2.800.000
                total += d * last_scale // 10
                last = "done"
            else:
                cur += d
                last = "digit"
        elif w == "mười":
            if last in ("digit", "tens", "tens_digit"):
                raise _Fail
            cur += 10
            last = "tens"
        elif w in ("mươi", "chục"):
            if last == "digit":
                d = cur % 10
                cur += d * 9  # d -> d*10
            elif last == "tens_digit":
                pass  # "hai trăm năm mươi": đã cộng 50 ở bước trước
            elif w == "chục" and last in (None, "scale"):
                cur += 10  # "chục triệu"
            else:
                raise _Fail
            last = "tens"
        elif w == "trăm":
            if last == "digit":
                d = cur % 10
                cur += d * 99
            elif last in (None, "scale"):
                cur += 100
            else:
                raise _Fail
            last = "trăm"
        elif w in ("linh", "lẻ"):
            if last != "trăm":
                raise _Fail
            last = "linh"
        elif w in SCALE:
            s = SCALE[w]
            if last_scale is not None and s > last_scale:
                total = (total + (cur or 1)) * s  # "một nghìn tỷ"
            else:
                total += (cur or 1) * s
            cur, last, last_scale = 0, "scale", s
        elif w == "rưỡi":
            if last == "scale":
                total += last_scale // 2
            elif last == "trăm":
                cur += 50
            else:
                raise _Fail
            last = "done"
        else:
            raise _Fail
        i += 1
    if last == "linh":
        raise _Fail
    return total + cur


def _split_struct_span(span: list[str]) -> list[list[str]]:
    """Một dãy từ số liền nhau có thể gồm nhiều số ("mười tám giờ hai mươi" đã bị tách bởi "giờ";
    nhưng "năm mươi hai mươi" thì không). Ở đây chỉ tách khi hai cụm hợp lệ nối nhau; thử từ trái sang."""
    out, start = [], 0
    while start < len(span):
        for end in range(len(span), start, -1):
            try:
                _parse_cardinal(span[start:end])
                out.append(span[start:end])
                start = end
                break
            except _Fail:
                continue
        else:
            out.append(span[start:start + 1])
            start += 1
    return out


def _convert(tokens: list[str], prev: str | None, nxt: str | None) -> str | None:
    """Một cụm từ số -> chuỗi chữ số, hoặc None nếu giữ nguyên chữ."""
    if all(t in DIGIT for t in tokens) and len(tokens) >= 4:
        return "".join(str(DIGIT[t]) for t in tokens)
    has_struct = any(t in STRUCT and t not in ("linh", "lẻ", "rưỡi") for t in tokens)
    if len(tokens) == 1 and tokens[0] in DIGIT:
        if tokens[0] == "không":
            return None
        if (nxt in UNITS) or (prev in PREFIX):
            return str(DIGIT[tokens[0]])
        return None
    if not has_struct:
        return None
    if len(tokens) == 1 and tokens[0] in ("chục", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ"):
        return None  # "vài chục", "hàng trăm"
    try:
        return str(_parse_cardinal(tokens))
    except _Fail:
        return None


def _compact(a: str, b: str, nxt: str | None) -> str | None:
    """Cách nói gọn 2 chữ số: "chín lăm" = 95, "hai mốt" = 21, "hai tư độ" = 24, "hai sáu độ" = 26.
    "hai ba ngày" (2-3 ngày) để nguyên: hai chữ số thường chỉ gộp khi đứng trước "độ"."""
    d = DIGIT.get(a)
    if d is None or d < 2:
        return None
    if b in ("lăm", "nhăm", "mốt") or (b == "tư" and nxt in UNITS):
        return str(d * 10 + UNIT_FORMS[b])
    if b in DIGIT and DIGIT[b] >= 1 and nxt == "độ":
        return str(d * 10 + DIGIT[b])
    return None


def itn(text: str) -> str:
    """Áp dụng trên văn bản đã normalize_asr (chữ thường, không dấu câu)."""
    toks = text.split()
    out, i = [], 0
    while i < len(toks):
        if i + 1 < len(toks):
            c = _compact(toks[i], toks[i + 1], toks[i + 2] if i + 2 < len(toks) else None)
            if c is not None:
                out.append(c)
                i += 2
                continue
        if toks[i] not in NUMBER_WORDS or toks[i] in ("linh", "lẻ", "rưỡi", "mốt", "tư", "lăm", "nhăm"):
            out.append(toks[i])
            i += 1
            continue
        j = i
        while j < len(toks) and toks[j] in NUMBER_WORDS:
            j += 1
        span = toks[i:j]
        # "rưỡi" chỉ thuộc cụm số khi đi sau nghìn/triệu/trăm; "năm rưỡi" bị tách thành "năm" + "rưỡi"
        pieces = [span] if all(t in DIGIT for t in span) else _split_struct_span(span)
        k = i
        for p in pieces:
            prev = toks[k - 1] if k > 0 else None
            nxt = toks[k + len(p)] if k + len(p) < len(toks) else None
            conv = _convert(p, prev, nxt)
            out += [conv] if conv is not None else p
            k += len(p)
        i = j
    return " ".join(out)


def llm_input(asr_text: str) -> str:
    return itn(normalize_asr(asr_text))
