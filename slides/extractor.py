"""
Slide content extractors for PPTX and PDF formats.
Each function returns a (title, content, speaker_notes) tuple.
"""
from typing import Tuple


def extract_pptx_slide(slide) -> Tuple[str, str, str]:
    """Extract text and speaker notes from a python-pptx slide object."""
    from pptx.util import Pt
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    title = ""
    content_parts = []

    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        text = shape.text_frame.text.strip()
        if not text:
            continue
        # Shape with "title" in its name or placeholder type 1 = title
        name_lower = shape.name.lower()
        if "title" in name_lower or (hasattr(shape, "placeholder_format") and
                                      shape.placeholder_format is not None and
                                      shape.placeholder_format.idx == 0):
            title = text
        else:
            content_parts.append(text)

    speaker_notes = ""
    if slide.has_notes_slide:
        try:
            notes_frame = slide.notes_slide.notes_text_frame
            speaker_notes = notes_frame.text.strip() if notes_frame else ""
        except Exception:
            pass

    return title, "\n".join(content_parts), speaker_notes


def extract_pdf_page(page) -> Tuple[str, str, str]:
    """Extract text from a PyMuPDF page object."""
    raw = page.get_text().strip()
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    title = lines[0] if lines else ""
    content = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return title, content, ""
