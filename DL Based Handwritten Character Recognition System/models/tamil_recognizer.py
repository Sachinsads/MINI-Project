"""
models/tamil_recognizer.py
──────────────────────────
Dedicated Tamil Handwritten Character Recognition Engine.

Selected model (best available for Tamil handwriting on HuggingFace):
  ┌─────────────────────────────────────────────────────────────────┐
  │ sabaridsnfuji/Tamil_Offline_Handwritten_OCR                     │
  │ Architecture: VisionEncoderDecoderModel                         │
  │   • Encoder: google/vit-base-patch16-224-in21k (ViT-Base)      │
  │   • Decoder: d42kw01f/Tamil-RoBERTa (Tamil-specific LM)        │
  │ Task: Offline handwritten Tamil text recognition                 │
  │ Why chosen: Only purpose-built Tamil handwriting TrOCR on HF;  │
  │   uses Tamil-specific language model as decoder for better       │
  │   Tamil Unicode character generation.                            │
  └─────────────────────────────────────────────────────────────────┘

Fallback chain:
  1. sabaridsnfuji/Tamil_Offline_Handwritten_OCR (Primary)
  2. EasyOCR Tamil CRNN (Windows-safe, secondary)
  3. Tesseract Tamil (tertiary, if installed)
"""

from __future__ import annotations

import io
import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# ── HuggingFace model IDs ─────────────────────────────────────────────────────
TAMIL_MODEL_ID    = "sabaridsnfuji/Tamil_Offline_Handwritten_OCR"
TAMIL_VIT_ID      = "google/vit-base-patch16-224-in21k"
TAMIL_ROBERTA_ID  = "d42kw01f/Tamil-RoBERTa"

# ── Lazy singletons ───────────────────────────────────────────────────────────
_model        = None
_processor    = None
_model_tried  = False     # True once we've attempted to load (success or fail)
_device       = None

_easy_reader  = None
_easy_tried   = False


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_device():
    global _device
    if _device is None:
        try:
            import torch
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            _device = None
    return _device


def _pil_to_rgb(img: Image.Image) -> Image.Image:
    """Ensure image is RGB with reasonable size for ViT (224×224 preferred)."""
    img = img.convert("RGB")
    w, h = img.size
    # ViT expects 224×224 but TrOCRProcessor handles resizing internally
    # If image is very tiny, upscale to help the ViT encoder
    if w < 32 or h < 32:
        scale = max(32 / w, 32 / h)
        img = img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.LANCZOS)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 1 — sabaridsnfuji/Tamil_Offline_Handwritten_OCR
# ─────────────────────────────────────────────────────────────────────────────

def _load_tamil_model() -> bool:
    """
    Load the Tamil VisionEncoderDecoder model.
    Uses the correct processor construction:
        TrOCRProcessor(
            image_processor = ViTImageProcessor.from_pretrained(VIT_ID),
            tokenizer       = AutoTokenizer.from_pretrained(ROBERTA_ID),
        )
    Returns True if loaded successfully.
    """
    global _model, _processor, _model_tried

    if _model_tried:
        return _model is not None

    _model_tried = True

    try:
        from transformers import (
            VisionEncoderDecoderModel,
            ViTImageProcessor,
            AutoFeatureExtractor,
            AutoTokenizer,
            TrOCRProcessor,
        )
        import torch

        device = _get_device()
        logger.info(f"[Tamil OCR] Loading {TAMIL_MODEL_ID} on {device}...")

        # ── Image processor (ViT feature extractor) ──
        try:
            feat = ViTImageProcessor.from_pretrained(TAMIL_VIT_ID)
        except Exception:
            feat = AutoFeatureExtractor.from_pretrained(TAMIL_VIT_ID)

        # ── Tokenizer (Tamil RoBERTa) ──
        tok = AutoTokenizer.from_pretrained(TAMIL_ROBERTA_ID)

        # ── Combine into TrOCRProcessor ──
        _processor = TrOCRProcessor(image_processor=feat, tokenizer=tok)

        # ── Load model weights ──
        _model = VisionEncoderDecoderModel.from_pretrained(TAMIL_MODEL_ID)
        if device is not None:
            _model = _model.to(device)
        _model.eval()

        logger.info("[Tamil OCR] ✓ sabaridsnfuji/Tamil_Offline_Handwritten_OCR loaded")
        return True

    except Exception as exc:
        logger.warning(f"[Tamil OCR] Model load failed: {exc}")
        _model = None
        _processor = None
        return False


def _run_trocr(pil_image: Image.Image) -> str:
    """Run the Tamil TrOCR model on a single PIL image."""
    if _processor is None or _model is None:
        return ""
    try:
        import torch

        device = _get_device()
        img = _pil_to_rgb(pil_image)

        pixel_values = _processor(
            images=img, return_tensors="pt"
        ).pixel_values

        if device is not None:
            pixel_values = pixel_values.to(device)

        with torch.no_grad():
            generated_ids = _model.generate(
                pixel_values,
                max_new_tokens=200,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    except Exception as exc:
        logger.warning(f"[Tamil OCR] TrOCR inference error: {exc}")
        return ""


def recognize_with_trocr(
    full_image: Image.Image,
    line_images: Optional[list[Image.Image]] = None,
) -> str:
    """
    Run Tamil TrOCR on the image(s).
    If line_images provided (multiple segmented lines), runs on each line
    and joins results. Otherwise runs on the full image.
    """
    if not _load_tamil_model():
        return ""

    # Multi-line: run on each segmented line crop
    if line_images and len(line_images) >= 1:
        results = []
        for i, limg in enumerate(line_images):
            text = _run_trocr(limg)
            if text:
                results.append(text)
                logger.debug(f"  [Tamil] Line {i+1}: {repr(text[:50])}")
        if results:
            combined = "\n".join(results)
            logger.info(f"[Tamil OCR] TrOCR line-by-line: {len(results)} line(s), {len(combined)} chars")
            return combined

    # Full image fallback
    text = _run_trocr(full_image)
    if text:
        logger.info(f"[Tamil OCR] TrOCR full-image: {len(text)} chars")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 2 — EasyOCR Tamil (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────

def _load_easyocr_tamil():
    """
    Load EasyOCR Tamil reader, suppressing Windows cp1252 console crash
    caused by block-character progress bars during model download.
    """
    global _easy_reader, _easy_tried
    if _easy_tried:
        return _easy_reader
    _easy_tried = True

    try:
        import easyocr

        # Suppress stdout/stderr during load to avoid Windows charset crash
        old_out, old_err = sys.stdout, sys.stderr
        devnull_out = open(os.devnull, "w", encoding="utf-8")
        devnull_err = open(os.devnull, "w", encoding="utf-8")
        try:
            sys.stdout = devnull_out
            sys.stderr = devnull_err
            _easy_reader = easyocr.Reader(["ta", "en"], gpu=False, verbose=False)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
            devnull_out.close()
            devnull_err.close()

        logger.info("[Tamil OCR] ✓ EasyOCR Tamil reader loaded (Windows-safe)")
        return _easy_reader

    except Exception as exc:
        logger.warning(f"[Tamil OCR] EasyOCR load failed: {exc}")
        return None


def recognize_with_easyocr(pil_image: Image.Image) -> str:
    """Run EasyOCR on the full image for Tamil."""
    reader = _load_easyocr_tamil()
    if reader is None:
        return ""
    try:
        img_np = np.array(pil_image.convert("RGB"))
        results = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(r.strip() for r in results if r.strip())
        if text:
            logger.info(f"[Tamil OCR] EasyOCR: {len(text)} chars")
        return text
    except Exception as exc:
        logger.warning(f"[Tamil OCR] EasyOCR inference error: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE 3 — Tesseract Tamil (optional)
# ─────────────────────────────────────────────────────────────────────────────

def recognize_with_tesseract(pil_image: Image.Image) -> str:
    """
    Use Tesseract with Tamil language pack as a tertiary fallback.
    Requires: pip install pytesseract  +  tesseract-ocr binary with tam langpack.
    """
    try:
        import pytesseract  # type: ignore
        img_gray = pil_image.convert("L")
        text = pytesseract.image_to_string(img_gray, lang="tam", config="--psm 6 --oem 1")
        text = text.strip()
        if text:
            logger.info(f"[Tamil OCR] Tesseract: {len(text)} chars")
        return text
    except Exception as exc:
        logger.debug(f"[Tamil OCR] Tesseract not available: {exc}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def recognize_tamil(
    full_pil_image: Image.Image,
    line_images: Optional[list[Image.Image]] = None,
) -> dict:
    """
    Run the full Tamil OCR cascade:
      1. sabaridsnfuji/Tamil_Offline_Handwritten_OCR  (primary)
      2. EasyOCR Tamil CRNN                            (secondary)
      3. Tesseract Tamil                               (tertiary)

    Returns a dict matching the main recognizer.py output format:
      { lines, full_text, language, engine_used }
    """
    text = ""
    engine_used = ""

    # ── Engine 1: Tamil TrOCR ──────────────────────────────────────
    logger.info("[Tamil OCR] → Engine 1: sabaridsnfuji/Tamil_Offline_Handwritten_OCR")
    text = recognize_with_trocr(full_pil_image, line_images)
    if text:
        engine_used = (
            "Tamil TrOCR (ViT + Tamil-RoBERTa) — Line-by-Line"
            if line_images and len(line_images) > 1
            else "Tamil TrOCR (ViT + Tamil-RoBERTa) — Full Image"
        )

    # ── Engine 2: EasyOCR Tamil ────────────────────────────────────
    if not text:
        logger.info("[Tamil OCR] → Engine 2: EasyOCR Tamil CRNN")
        text = recognize_with_easyocr(full_pil_image)
        if text:
            engine_used = "EasyOCR Tamil CRNN (Fallback)"

    # ── Engine 3: Tesseract ────────────────────────────────────────
    if not text:
        logger.info("[Tamil OCR] → Engine 3: Tesseract Tamil")
        text = recognize_with_tesseract(full_pil_image)
        if text:
            engine_used = "Tesseract Tamil (Fallback)"

    # ── All engines failed ─────────────────────────────────────────
    if not text:
        logger.warning("[Tamil OCR] All engines returned empty output.")
        engine_used = "Tamil OCR — No output detected"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return {
        "lines": lines,
        "full_text": text,
        "language": "tamil",
        "engine_used": engine_used,
    }


def preload() -> None:
    """Eagerly load the Tamil TrOCR model (call at startup to reduce first-request latency)."""
    logger.info("[Tamil OCR] Pre-loading Tamil TrOCR model...")
    _load_tamil_model()
    logger.info("[Tamil OCR] Pre-load complete.")
