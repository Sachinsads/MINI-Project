"""
models/model_loader.py
──────────────────────
State-of-the-art OCR Engine Loader:
  • Primary Engine:   Google Gemini 3.6 Flash Vision AI (Multimodal Vision API with User Key)
  • Local Suite:      Microsoft TrOCR (Base / Small / Large), PARSeq (baudm/parseq), EasyOCR (Indic CRNN)
"""
from __future__ import annotations

import logging
import torch
import easyocr
from transformers import ViTImageProcessor, RobertaTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)

# Active API Key provided by user
GEMINI_API_KEY = "AQ.Ab8RN6KZPS3Uqfb_1f88k-BeGsRpeu7tGJSs-vqxOiyQ_R96Hg"

# Flagship Primary Vision Model ID list for multi-model failover
PRIMARY_MODEL_ID = "gemini-3.6-flash"
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

MODEL_CONFIGS = {
    "english": {
        "label": "Multimodal Vision Engine + TrOCR Suite (English)",
        "gemini_model": PRIMARY_MODEL_ID,
        "easyocr_langs": ["en"],
        "description": "Multimodal Vision Cloud Engine + Microsoft TrOCR (Base/Small/Large) & PARSeq",
        "size_mb": 650,
    },
    "hindi": {
        "label": "Multimodal Vision Engine + EasyOCR (Hindi Devanagari)",
        "gemini_model": PRIMARY_MODEL_ID,
        "easyocr_langs": ["hi", "en"],
        "description": "Multimodal Vision Cloud Engine + EasyOCR Devanagari CRNN model",
        "size_mb": 120,
    },
    "tamil": {
        "label": "Tamil TrOCR (ViT+Tamil-RoBERTa) + Multimodal Vision Engine",
        "gemini_model": PRIMARY_MODEL_ID,
        "easyocr_langs": ["ta", "en"],
        "description": "sabaridsnfuji/Tamil_Offline_Handwritten_OCR (ViT encoder + Tamil-RoBERTa decoder) with Multimodal Vision Engine primary and EasyOCR fallback",
        "size_mb": 480,
    },
}

_registry: dict = {}
_device = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    return _device


def load_model(lang: str) -> dict:
    if lang in _registry:
        return _registry[lang]

    if lang not in MODEL_CONFIGS:
        raise ValueError(f"Unknown language '{lang}'. Choose from: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[lang]
    device = get_device()
    use_gpu = torch.cuda.is_available()

    # Load EasyOCR fallback reader (Windows-safe: suppress UTF-8 progress bar)
    easyocr_reader = None
    if lang != "tamil":  # Tamil uses dedicated tamil_recognizer.py engine
        try:
            import sys, os as _os
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout = open(_os.devnull, "w", encoding="utf-8")
                sys.stderr = open(_os.devnull, "w", encoding="utf-8")
                easyocr_reader = easyocr.Reader(cfg["easyocr_langs"], gpu=use_gpu, verbose=False)
            finally:
                sys.stdout.close(); sys.stderr.close()
                sys.stdout, sys.stderr = old_stdout, old_stderr
        except Exception as e:
            logger.debug(f"EasyOCR fallback notice [{lang}]: {e}")

    # Load TrOCR Base (Default Local English Model)
    trocr_processor = None
    trocr_model = None
    try:
        shared_key = "__shared_trocr_base__"
        if shared_key in _registry:
            trocr_processor = _registry[shared_key]["processor"]
            trocr_model = _registry[shared_key]["model"]
        else:
            ip = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            tok = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
            trocr_processor = TrOCRProcessor(image_processor=ip, tokenizer=tok)
            trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            trocr_model.to(device)
            trocr_model.eval()
            _registry[shared_key] = {"processor": trocr_processor, "model": trocr_model}
    except Exception as e:
        logger.debug(f"TrOCR Base notice: {e}")

    # Load PARSeq (baudm/parseq) via PyTorch Hub if available
    parseq_model = None
    try:
        shared_parseq = "__shared_parseq__"
        if shared_parseq in _registry:
            parseq_model = _registry[shared_parseq]["model"]
        else:
            parseq_model = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(device)
            _registry[shared_parseq] = {"model": parseq_model}
    except Exception as e:
        logger.debug(f"PARSeq notice: {e}")

    entry = {
        "gemini_key": GEMINI_API_KEY,
        "gemini_model": PRIMARY_MODEL_ID,
        "easyocr": easyocr_reader,
        "trocr_processor": trocr_processor,
        "trocr_model": trocr_model,
        "parseq_model": parseq_model,
        "device": device,
        "config": cfg,
    }
    _registry[lang] = entry
    return entry


def load_all_models() -> None:
    for lang in MODEL_CONFIGS:
        try:
            load_model(lang)
        except Exception as e:
            logger.warning(f"Loader error for {lang}: {e}")


def get_loaded_languages() -> list:
    return [k for k in _registry if not k.startswith("__shared")]

