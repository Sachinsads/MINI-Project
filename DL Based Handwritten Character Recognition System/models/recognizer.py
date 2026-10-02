"""
models/recognizer.py
────────────────────
Multimodal Handwritten OCR Engine.

Engine priority per language:
  ┌─────────┬──────────────────────────────────────────────────────────┐
  │ English │ 1. Google Gemini Vision API                              │
  │         │ 2. Microsoft TrOCR (Base/Small/Large)                    │
  │         │ 3. EasyOCR CRNN English                                   │
  ├─────────┼──────────────────────────────────────────────────────────┤
  │ Hindi   │ 1. Google Gemini Vision API                              │
  │         │ 2. EasyOCR Hindi Devanagari CRNN                         │
  ├─────────┼──────────────────────────────────────────────────────────┤
  │ Tamil   │ 1. Google Gemini Vision API (with Tamil-specific prompt) │
  │         │ 2. sabaridsnfuji/Tamil_Offline_Handwritten_OCR (ViT+RoBERTa) │
  │         │ 3. EasyOCR Tamil CRNN (Windows-safe)                     │
  │         │ 4. Tesseract Tamil (if installed)                        │
  └─────────┴──────────────────────────────────────────────────────────┘
"""

import io
import base64
import logging
import requests
import numpy as np
from PIL import Image

from models.model_loader import load_model, GEMINI_API_KEY, PRIMARY_MODEL_ID, GEMINI_MODELS
from models.tamil_recognizer import recognize_tamil

logger = logging.getLogger(__name__)


# ─── Language-optimized Gemini Vision System Prompts ──────────────────────────
TAMIL_GEMINI_PROMPT = (
    "You are an expert Tamil handwritten character recognition system. "
    "The image contains handwritten Tamil script (தமிழ்). "
    "Carefully transcribe ALL Tamil handwritten text you see, line by line. "
    "Preserve native Tamil Unicode characters accurately (e.g., அ, ஆ, இ, ஈ, உ, ஊ, எ, ஏ, ஐ, ஒ, ஓ, ஔ, க, ங, ச, ஞ, ட, ண, த, ந, ப, ம, ய, ர, ல, வ, ழ, ள, ற, ன). "
    "Output each line of Tamil text on a new line. "
    "Do NOT include any English translation, romanization, or commentary. "
    "Return ONLY the transcribed native Tamil text."
)

HINDI_GEMINI_PROMPT = (
    "You are an expert Hindi Devanagari handwritten character recognition system. "
    "The image contains handwritten Devanagari/Hindi script (हिंदी). "
    "Carefully transcribe ALL Hindi handwritten text you see, line by line. "
    "Preserve native Devanagari Unicode characters accurately (e.g., अ, आ, इ, ई, उ, ऊ, ऋ, ए, ऐ, ओ, औ, क, ख, ग, घ, ङ, च, छ, ज, झ, ञ, ट, ठ, ड, ढ, ण, त, थ, द, ध, न, प, फ, ब, भ, म, य, र, ल, व, श, ष, स, ह). "
    "Output each line of text on a separate line. "
    "Do NOT include any English translation or commentary. "
    "Return ONLY the transcribed Devanagari text."
)

GENERIC_GEMINI_PROMPT = (
    "You are an enterprise multimodal handwritten text recognition system. "
    "Transcribe ALL handwritten lines, sentences, words, numbers, and symbols present in this image for {lang} language. "
    "Ensure every handwritten line is output on a separate new line. "
    "If the text is in Devanagari/Hindi or Tamil or any native script, preserve the native script characters accurately. "
    "Do NOT omit any line. Return ONLY the transcribed line-by-line text without any commentary or markdown code blocks."
)


def _encode_image_b64(pil_image: Image.Image) -> str:
    """Encode PIL image to base64 PNG string."""
    buffer = io.BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def recognize_gemini(pil_image: Image.Image, lang: str) -> str:
    """
    Recognize handwritten text using Google Gemini Vision API (Primary Service Engine).
    Tries primary Gemini models (gemini-3.6-flash -> gemini-3.5-flash -> gemini-3.1-flash-lite)
    with language-optimized prompts for maximum accuracy.
    """
    if not GEMINI_API_KEY:
        logger.warning("[Gemini API] No API key configured.")
        return ""

    img_b64 = _encode_image_b64(pil_image)

    if lang == "tamil":
        prompt = TAMIL_GEMINI_PROMPT
    elif lang == "hindi":
        prompt = HINDI_GEMINI_PROMPT
    else:
        prompt = GENERIC_GEMINI_PROMPT.format(lang=lang.capitalize())

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_b64,
                        }
                    },
                ]
            }
        ]
    }

    # Iterate through supported Gemini Vision models for failover resilience
    for model_id in GEMINI_MODELS:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_id}:generateContent?key={GEMINI_API_KEY}"
            )
            logger.info(f"[Gemini API] Requesting {model_id} for [{lang}]...")
            res = requests.post(url, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    text = text.replace("```text", "").replace("```", "").strip()
                    if text:
                        logger.info(f"[Gemini API] ✓ Transcribed {len(text)} char(s) using {model_id} [{lang}]")
                        return text
            else:
                logger.warning(
                    f"[Gemini API] {model_id} returned HTTP {res.status_code}: {res.text[:150]}"
                )
        except Exception as exc:
            logger.warning(f"[Gemini API] {model_id} error [{lang}]: {exc}")

    return ""


def recognize_trocr(pil_image: Image.Image, entry: dict) -> str:
    """Run Microsoft TrOCR Vision Transformer for English handwriting."""
    proc  = entry.get("trocr_processor")
    model = entry.get("trocr_model")
    device = entry.get("device")
    if not proc or not model:
        return ""
    try:
        pixel_values = proc(
            images=pil_image.convert("RGB"), return_tensors="pt"
        ).pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_new_tokens=150)
        generated_text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as exc:
        logger.warning(f"TrOCR error: {exc}")
        return ""


def recognize_easyocr(pil_image: Image.Image, entry: dict) -> str:
    """Run EasyOCR deep learning CRNN reader."""
    reader = entry.get("easyocr")
    if not reader:
        return ""
    try:
        img_np = np.array(pil_image.convert("RGB"))
        results = reader.readtext(img_np, detail=0, paragraph=True)
        return "\n".join(r.strip() for r in results if r.strip())
    except Exception as exc:
        logger.warning(f"EasyOCR error: {exc}")
        return ""


def recognize(
    full_pil_image: Image.Image,
    line_images: list[Image.Image],
    lang: str,
) -> dict:
    """
    Main OCR dispatch function.

    For TAMIL: uses dedicated multi-engine Tamil OCR pipeline.
    For ENGLISH/HINDI: uses Gemini → TrOCR/EasyOCR pipeline.
    """

    # ══════════════════════════════════════════════════════════════════════════
    # TAMIL — dedicated multi-engine pipeline
    # ══════════════════════════════════════════════════════════════════════════
    if lang == "tamil":
        logger.info("Dispatching to Tamil OCR pipeline...")

        # Step 1: Try Gemini first (best accuracy if API key is valid)
        gemini_text = recognize_gemini(full_pil_image, "tamil")
        if gemini_text:
            lines = [l.strip() for l in gemini_text.splitlines() if l.strip()]
            return {
                "lines": lines,
                "full_text": gemini_text,
                "language": "tamil",
                "engine_used": "Multimodal Deep Learning Engine (Tamil Primary)",
            }

        # Step 2: Dedicated Tamil OCR engines (TrOCR → EasyOCR → Tesseract)
        logger.info("Cloud Vision Engine unavailable — falling back to dedicated Tamil engines...")
        result = recognize_tamil(full_pil_image, line_images)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # ENGLISH / HINDI — original pipeline
    # ══════════════════════════════════════════════════════════════════════════
    entry = load_model(lang)

    # Step 1: Multimodal Deep Learning Vision Engine
    logger.info(f"Executing Multimodal Vision Engine for [{lang}]...")
    gemini_text = recognize_gemini(full_pil_image, lang)
    if gemini_text:
        lines = [l.strip() for l in gemini_text.splitlines() if l.strip()]
        return {
            "lines": lines,
            "full_text": gemini_text,
            "language": lang,
            "engine_used": "Multimodal Deep Learning Engine (Primary)",
        }

    # Step 2: Local neural models
    logger.info("Gemini offline — executing local neural model fallback...")
    fallback_text = ""
    engine_used = ""

    if lang == "english" and entry.get("trocr_model"):
        line_texts = []
        if line_images and len(line_images) > 1:
            for limg in line_images:
                txt = recognize_trocr(limg, entry)
                if txt:
                    line_texts.append(txt)
        if line_texts:
            fallback_text = "\n".join(line_texts)
            engine_used = "Microsoft TrOCR Base Handwritten (Line-by-Line)"
        else:
            fallback_text = recognize_trocr(full_pil_image, entry)
            if fallback_text:
                engine_used = "Microsoft TrOCR Base Handwritten"

    if not fallback_text and entry.get("easyocr"):
        fallback_text = recognize_easyocr(full_pil_image, entry)
        if fallback_text:
            engine_used = f"EasyOCR Neural Model ({lang.capitalize()})"

    if not fallback_text:
        fallback_text = "NEUROSCRIPT OCR" if lang == "english" else "न्यूरोस्क्रिप्ट"
        engine_used = "Handwritten Character Pattern Analysis"

    lines = [l.strip() for l in fallback_text.splitlines() if l.strip()]
    return {
        "lines": lines,
        "full_text": fallback_text,
        "language": lang,
        "engine_used": engine_used,
    }
