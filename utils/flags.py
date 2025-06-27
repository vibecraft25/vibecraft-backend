__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"


def parse_flags_from_text(text: str) -> tuple[bool, bool]:
    text = text.lower()
    redo = any(k in text for k in ["다시", "redo", "변경", "재시도"])
    go_back = any(k in text for k in ["이전", "undo", "되돌리기"])
    return redo, go_back
