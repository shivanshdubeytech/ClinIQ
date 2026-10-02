"""Stage A: Document and Image Text Extractor with Confidence Scoring.

Extracts text from medical documents (digital PDFs and images).
Calculates per-item and overall extraction confidence scores,
evaluates image sharpness/contrast for blur detection, and ensures
safe cleanup of temporary uploaded files according to privacy constraints.
"""

from dataclasses import dataclass, field
import io
import math
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union
import site
import sys

def _ensure_pdf_reader():
    """Dynamically locates and imports PdfReader from user or system packages."""
    global PdfReader
    if PdfReader is not None:
        return PdfReader

    try:
        from pypdf import PdfReader as _pr
        PdfReader = _pr
        return PdfReader
    except ImportError:
        pass

    candidates = [
        site.getusersitepackages(),
        os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages"),
        os.path.expanduser(r"~\AppData\Roaming\Python\Python39\site-packages"),
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python39\Lib\site-packages"),
    ]
    for p in candidates:
        if p and p not in sys.path and Path(p).exists():
            sys.path.append(str(p))

    try:
        from pypdf import PdfReader as _pr
        PdfReader = _pr
        return PdfReader
    except ImportError:
        PdfReader = None
        return None

# Auto-discover user site-packages
PdfReader = None
_ensure_pdf_reader()

try:
    from PIL import Image, ImageFile, ImageStat
    from PIL.Image import DecompressionBombError
    # Cap maximum image pixels to 50 million to prevent decompression bombs / OOM crashes
    Image.MAX_IMAGE_PIXELS = 50_000_000
    ImageFile.LOAD_TRUNCATED_IMAGES = False
except ImportError:
    Image = None
    ImageFile = None
    ImageStat = None
    DecompressionBombError = Exception

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB ceiling for medical documents




@dataclass
class ExtractedItem:
    """Represents a single extracted line or text segment with its confidence score."""
    text: str
    confidence: float
    source_type: str  # "pdf_digital", "image_ocr", "text"
    page_num: int = 1
    line_num: int = 1

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "source_type": self.source_type,
            "page_num": self.page_num,
            "line_num": self.line_num,
        }


@dataclass
class ExtractionResult:
    """Encapsulates the complete extraction outcome for a medical document."""
    raw_text: str
    items: List[ExtractedItem] = field(default_factory=list)
    overall_confidence: float = 0.0
    document_type: str = "unknown"  # "pdf", "image", "text", "empty", "invalid"
    page_count: int = 0
    is_blurry_or_low_quality: bool = False
    quality_notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        return self.error is None and len(self.raw_text.strip()) > 0

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "overall_confidence": round(self.overall_confidence, 3),
            "document_type": self.document_type,
            "page_count": self.page_count,
            "is_blurry_or_low_quality": self.is_blurry_or_low_quality,
            "quality_notes": self.quality_notes,
            "item_count": len(self.items),
            "error": self.error,
        }


def assess_image_quality(image_input: Union[Path, str, io.BytesIO, "Image.Image"]) -> Tuple[bool, float, List[str]]:
    """Evaluates image resolution, contrast, and brightness to detect blurry or degraded reports.

    Returns:
        Tuple of (is_blurry_or_low_quality, quality_score, notes)
    """
    if Image is None:
        return False, 0.7, ["Pillow library not available for visual quality metrics."]

    notes = []
    try:
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input)
        elif isinstance(image_input, io.BytesIO):
            img = Image.open(image_input)
        else:
            img = image_input

        # Convert to grayscale for statistical analysis
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)

        width, height = img.size
        # Minimum resolution check for medical documents
        if width < 400 or height < 400:
            notes.append(f"Low resolution: {width}x{height} pixels.")

        # Contrast assessment via standard deviation of pixel intensities
        std_dev = stat.stddev[0]
        if std_dev < 20:
            notes.append(f"Extremely low contrast (stddev: {std_dev:.1f}). Text may be washed out.")
        elif std_dev < 35:
            notes.append(f"Suboptimal contrast (stddev: {std_dev:.1f}).")

        # Mean brightness assessment (0 = all black, 255 = all white)
        mean_brightness = stat.mean[0]
        if mean_brightness < 40:
            notes.append("Image is too dark to reliably distinguish characters.")
        elif mean_brightness > 245:
            notes.append("Image is overexposed or nearly blank.")

        # Estimate sharpness via bounded strided point-sampling (O(1) memory, avoids large pixel list allocation)
        pixel_count = width * height
        if pixel_count > 1000:
            sample_target = min(5000, pixel_count)
            step_x = max(1, width // int(math.isqrt(sample_target) or 1))
            step_y = max(1, height // int(math.isqrt(sample_target) or 1))
            diff_sum = 0
            sample_count = 0
            for y in range(0, height - 1, step_y):
                for x in range(0, width - 1, step_x):
                    p1 = gray.getpixel((x, y))
                    p2 = gray.getpixel((x + 1, y))
                    diff_sum += abs(p1 - p2)
                    sample_count += 1
                    if sample_count >= 5000:
                        break
                if sample_count >= 5000:
                    break
            avg_gradient = diff_sum / max(1, sample_count)
            if avg_gradient < 3.0:
                notes.append(f"Image appears heavily blurred or uniform (gradient: {avg_gradient:.1f}).")

        is_low_quality = len(notes) > 0
        quality_score = 0.95
        if notes:
            quality_score = max(0.2, 0.95 - (0.25 * len(notes)))

        return is_low_quality, quality_score, notes

    except DecompressionBombError as dbe:
        return True, 0.0, [f"Image exceeds maximum allowable pixel dimension (possible decompression bomb): {dbe}"]
    except Exception as err:
        return True, 0.3, [f"Failed to assess image quality: {err}"]


def _calculate_line_confidence(line_text: str, base_confidence: float = 1.0) -> float:
    """Calculates confidence score based on printable characters, medical formatting, and noise."""
    if not line_text.strip():
        return 0.0

    total_chars = len(line_text)
    if total_chars == 0:
        return 0.0

    # Ratio of standard alphanumeric / punctuation chars
    valid_chars = sum(1 for c in line_text if c.isprintable() and ord(c) < 128)
    char_validity_ratio = valid_chars / total_chars

    # Penalize strange non-printable or replacement character clusters (e.g.  or random garbage)
    garbage_matches = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", line_text))
    penalty = (garbage_matches * 0.2)

    confidence = (base_confidence * char_validity_ratio) - penalty
    return max(0.0, min(1.0, confidence))


def extract_from_pdf(pdf_path: Path) -> ExtractionResult:
    """Extracts digital text from a PDF file page by page, computing confidence per line."""
    reader_cls = _ensure_pdf_reader()
    if reader_cls is None:
        return ExtractionResult(
            raw_text="",
            error="pypdf is required for PDF processing.",
            document_type="pdf",
        )

    try:
        reader = reader_cls(str(pdf_path))
        page_count = len(reader.pages)

        if page_count == 0:
            return ExtractionResult(
                raw_text="",
                page_count=0,
                document_type="pdf",
                error="PDF document contains zero pages.",
            )

        extracted_items: List[ExtractedItem] = []
        raw_text_parts: List[str] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            raw_text_parts.append(page_text)
            lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
            for line_idx, line in enumerate(lines, start=1):
                conf = _calculate_line_confidence(line, base_confidence=0.98)
                extracted_items.append(
                    ExtractedItem(
                        text=line,
                        confidence=conf,
                        source_type="pdf_digital",
                        page_num=page_idx,
                        line_num=line_idx,
                    )
                )

        full_raw_text = "\n".join(raw_text_parts).strip()

        if not full_raw_text:
            return ExtractionResult(
                raw_text="",
                page_count=page_count,
                document_type="pdf",
                is_blurry_or_low_quality=True,
                quality_notes=["Digital text layer is empty. The PDF may be a scanned image."],
                overall_confidence=0.1,
                error=None,
            )

        avg_conf = (
            sum(item.confidence for item in extracted_items) / len(extracted_items)
            if extracted_items else 0.0
        )

        return ExtractionResult(
            raw_text=full_raw_text,
            items=extracted_items,
            overall_confidence=avg_conf,
            document_type="pdf",
            page_count=page_count,
        )

    except Exception as err:
        return ExtractionResult(
            raw_text="",
            error=f"Error parsing PDF: {str(err)}",
            document_type="pdf",
        )


def extract_from_image(image_path: Path) -> ExtractionResult:
    """Extracts text from an image file.

    Evaluates image blur/contrast quality, attempts OCR via pytesseract if installed,
    and returns granular confidence scoring.
    """
    if Image is None:
        return ExtractionResult(
            raw_text="",
            error="Pillow library is required for image extraction.",
            document_type="image",
        )

    is_blurry, quality_score, quality_notes = assess_image_quality(image_path)

    # Check for pytesseract availability
    try:
        import pytesseract
        has_tesseract = True
    except ImportError:
        has_tesseract = False

    if not has_tesseract:
        # Fallback when pytesseract is not yet installed:
        # Provide clean structured quality review and state OCR requirement cleanly
        note_msg = (
            "OCR engine (pytesseract) is not installed. Listed in requirements_new.txt."
            if not is_blurry
            else "Image is low quality/blurry and OCR engine is not installed."
        )
        quality_notes.append(note_msg)
        return ExtractionResult(
            raw_text="",
            overall_confidence=0.2 if is_blurry else 0.4,
            document_type="image",
            page_count=1,
            is_blurry_or_low_quality=is_blurry,
            quality_notes=quality_notes,
            error="OCR engine not installed. Add pytesseract to extract text from images.",
        )

    try:
        img = Image.open(str(image_path))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        extracted_items = []
        raw_lines = []
        current_line_words = []
        current_line_confs = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf_val = float(data["conf"][i])
            if word:
                norm_conf = max(0.0, min(1.0, conf_val / 100.0)) * quality_score
                current_line_words.append(word)
                current_line_confs.append(norm_conf)

            # When line ends or last box
            if (i == n_boxes - 1 or data["line_num"][i] != data["line_num"][min(i + 1, n_boxes - 1)]) and current_line_words:
                line_str = " ".join(current_line_words)
                avg_line_conf = sum(current_line_confs) / len(current_line_confs)
                extracted_items.append(
                    ExtractedItem(
                        text=line_str,
                        confidence=avg_line_conf,
                        source_type="image_ocr",
                        page_num=1,
                        line_num=len(extracted_items) + 1,
                    )
                )
                raw_lines.append(line_str)
                current_line_words = []
                current_line_confs = []

        full_text = "\n".join(raw_lines).strip()
        avg_conf = (
            sum(item.confidence for item in extracted_items) / len(extracted_items)
            if extracted_items else 0.0
        )

        return ExtractionResult(
            raw_text=full_text,
            items=extracted_items,
            overall_confidence=avg_conf,
            document_type="image",
            page_count=1,
            is_blurry_or_low_quality=is_blurry,
            quality_notes=quality_notes,
        )

    except DecompressionBombError as dbe:
        return ExtractionResult(
            raw_text="",
            overall_confidence=0.0,
            document_type="image",
            page_count=1,
            is_blurry_or_low_quality=True,
            quality_notes=["Image exceeds maximum allowable pixel dimension."],
            error=f"Image exceeds maximum safe dimensions (decompression bomb protection): {dbe}",
        )
    except Exception as err:
        return ExtractionResult(
            raw_text="",
            overall_confidence=0.0,
            document_type="image",
            page_count=1,
            is_blurry_or_low_quality=is_blurry,
            quality_notes=quality_notes,
            error=f"OCR execution failed: {str(err)}",
        )


def extract_medical_document(
    file_path: Union[str, Path],
    auto_delete_temp: bool = False,
) -> ExtractionResult:
    """Primary entrypoint for extracting text from medical files.

    Args:
        file_path: Absolute or relative path to the uploaded file.
        auto_delete_temp: When True, deletes the uploaded file after processing (Rule 8 privacy).

    Returns:
        ExtractionResult containing raw text, structured items, and confidence metrics.
    """
    path = Path(file_path)

    try:
        # 1. Existence check
        if not path.exists():
            return ExtractionResult(
                raw_text="",
                error=f"File does not exist: {path.name}",
                document_type="invalid",
            )

        # 2. File size check (Empty or oversized)
        file_size = path.stat().st_size
        if file_size == 0:
            return ExtractionResult(
                raw_text="",
                error="Uploaded file is empty (0 bytes).",
                document_type="empty",
            )
        if file_size > MAX_UPLOAD_BYTES:
            return ExtractionResult(
                raw_text="",
                error=f"Uploaded file exceeds maximum allowable size (20 MB). Provided size: {file_size / (1024*1024):.1f} MB.",
                document_type="invalid",
            )

        suffix = path.suffix.lower()

        # 3. PDF handling
        if suffix == ".pdf":
            return extract_from_pdf(path)

        # 4. Image handling
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}:
            return extract_from_image(path)

        # 5. Plain text support for structured report transcripts
        elif suffix in {".txt", ".csv"}:
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
                lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                items = [
                    ExtractedItem(
                        text=ln,
                        confidence=_calculate_line_confidence(ln, base_confidence=1.0),
                        source_type="text",
                        page_num=1,
                        line_num=idx,
                    )
                    for idx, ln in enumerate(lines, start=1)
                ]
                avg_conf = sum(it.confidence for it in items) / len(items) if items else 0.0
                return ExtractionResult(
                    raw_text=content,
                    items=items,
                    overall_confidence=avg_conf,
                    document_type="text",
                    page_count=1,
                )
            except Exception as err:
                return ExtractionResult(
                    raw_text="",
                    error=f"Failed to read text file: {err}",
                    document_type="text",
                )

        # 6. Unsupported / wrong file types
        else:
            return ExtractionResult(
                raw_text="",
                error=f"Unsupported file format '{suffix}'. Allowed: PDF, PNG, JPG, JPEG, WEBP, TXT.",
                document_type="invalid",
            )

    finally:
        # Rule 8: Never store or log uploaded medical files permanently.
        if auto_delete_temp and path.exists():
            try:
                path.unlink()
            except OSError:
                pass
