"""
Slide Engine — loads PPTX or PDF files, navigates slides, exports preview images.
"""
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SlideData:
    index: int          # 0-based
    total: int
    title: str
    content: str
    speaker_notes: str
    image_path: Optional[str] = None

    @property
    def number(self) -> int:
        """1-based slide number for display."""
        return self.index + 1

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "number": self.number,
            "total": self.total,
            "title": self.title,
            "content": self.content,
            "speaker_notes": self.speaker_notes,
            "image_path": self.image_path,
        }


class SlideManager:
    def __init__(self):
        self._slides: List[SlideData] = []
        self._current_index: int = 0
        self._source_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, file_path: str) -> int:
        """Load a PPTX or PDF file. Returns the slide count."""
        self._source_path = file_path
        self._slides = []
        self._current_index = 0

        ext = Path(file_path).suffix.lower()
        if ext in (".pptx", ".ppt"):
            self._load_pptx(file_path)
        elif ext == ".pdf":
            self._load_pdf(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use PPTX or PDF.")

        # Backfill total count
        total = len(self._slides)
        for slide in self._slides:
            slide.total = total

        logger.info("Loaded %d slides from %s", total, file_path)
        return total

    def _load_pptx(self, path: str):
        from pptx import Presentation
        from slides.extractor import extract_pptx_slide

        prs = Presentation(path)
        for i, slide in enumerate(prs.slides):
            title, content, notes = extract_pptx_slide(slide)
            self._slides.append(SlideData(
                index=i,
                total=0,
                title=title or f"Slide {i + 1}",
                content=content,
                speaker_notes=notes,
            ))

    def _load_pdf(self, path: str):
        import fitz  # PyMuPDF
        from slides.extractor import extract_pdf_page

        doc = fitz.open(path)
        for i in range(len(doc)):
            title, content, notes = extract_pdf_page(doc[i])
            self._slides.append(SlideData(
                index=i,
                total=0,
                title=title or f"Slide {i + 1}",
                content=content,
                speaker_notes=notes,
            ))
        doc.close()

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    @property
    def total_slides(self) -> int:
        return len(self._slides)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def is_at_end(self) -> bool:
        return self._current_index >= len(self._slides) - 1

    @property
    def is_at_start(self) -> bool:
        return self._current_index == 0

    def get_current(self) -> Optional[SlideData]:
        if not self._slides:
            return None
        return self._slides[self._current_index]

    def next(self) -> Optional[SlideData]:
        if not self.is_at_end:
            self._current_index += 1
        return self.get_current()

    def previous(self) -> Optional[SlideData]:
        if not self.is_at_start:
            self._current_index -= 1
        return self.get_current()

    def goto(self, index: int) -> Optional[SlideData]:
        """Jump to a 0-based slide index (clamps to valid range)."""
        self._current_index = max(0, min(index, len(self._slides) - 1))
        return self.get_current()

    def get_slide(self, index: int) -> Optional[SlideData]:
        if 0 <= index < len(self._slides):
            return self._slides[index]
        return None

    def all_slides(self) -> List[SlideData]:
        return list(self._slides)

    # ------------------------------------------------------------------ #
    # Image Export
    # ------------------------------------------------------------------ #

    def export_slide_image(self, index: int, output_dir: str) -> Optional[str]:
        """Render slide to PNG and return the file path (or None on failure)."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"slide_{index:03d}.png")

        if os.path.exists(output_path):
            return output_path

        ext = Path(self._source_path).suffix.lower() if self._source_path else ""

        if ext == ".pdf":
            return self._render_pdf_page(self._source_path, index, output_path)
        elif ext in (".pptx", ".ppt"):
            # Convert PPTX → PDF via LibreOffice, then render
            pdf_path = self._pptx_to_pdf(self._source_path)
            if pdf_path:
                return self._render_pdf_page(pdf_path, index, output_path)
        return None

    def _render_pdf_page(self, pdf_path: str, page_index: int, output_path: str) -> Optional[str]:
        try:
            import fitz
            doc = fitz.open(pdf_path)
            if page_index >= len(doc):
                return None
            page = doc[page_index]
            # 2× zoom for crisp 1920-wide renders
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            pix.save(output_path)
            doc.close()
            return output_path
        except Exception as exc:
            logger.warning("PDF render failed for page %d: %s", page_index, exc)
            return None

    def _pptx_to_pdf(self, pptx_path: str) -> Optional[str]:
        """Convert a PPTX to PDF using LibreOffice (headless). Returns PDF path."""
        pdf_path = pptx_path.replace(".pptx", ".pdf").replace(".ppt", ".pdf")
        if os.path.exists(pdf_path):
            return pdf_path
        try:
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to", "pdf",
                    "--outdir", os.path.dirname(pptx_path),
                    pptx_path,
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return pdf_path if os.path.exists(pdf_path) else None
        except Exception as exc:
            logger.warning("LibreOffice conversion failed: %s", exc)
            return None
