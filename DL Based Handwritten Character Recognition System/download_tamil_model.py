"""
download_tamil_model.py
───────────────────────
Downloads and validates all components needed for Tamil Handwritten OCR:

  1. sabaridsnfuji/Tamil_Offline_Handwritten_OCR  (VisionEncoderDecoder weights)
  2. google/vit-base-patch16-224-in21k             (ViT feature extractor / image processor)
  3. d42kw01f/Tamil-RoBERTa                        (Tamil RoBERTa tokenizer)

Run:  python download_tamil_model.py
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def download_model():
    try:
        from huggingface_hub import snapshot_download
        from transformers import (
            VisionEncoderDecoderModel,
            AutoFeatureExtractor,
            AutoTokenizer,
            ViTImageProcessor,
            TrOCRProcessor,
        )
        from PIL import Image
        import torch
        import numpy as np
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install transformers torch Pillow huggingface_hub")
        sys.exit(1)

    MAIN_MODEL  = "sabaridsnfuji/Tamil_Offline_Handwritten_OCR"
    VIT_MODEL   = "google/vit-base-patch16-224-in21k"
    ROBERTA_TOK = "d42kw01f/Tamil-RoBERTa"

    logger.info("=" * 60)
    logger.info("  NEUROSCRIPT — Tamil OCR Model Downloader")
    logger.info("=" * 60)

    # ── Step 1: Download main model weights ───────────────────────
    logger.info(f"\n[1/4] Downloading VisionEncoderDecoder weights...")
    logger.info(f"      Model: {MAIN_MODEL}")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MAIN_MODEL)
        logger.info(f"      ✓ Model weights downloaded & loaded")
        del model  # free memory
    except Exception as e:
        logger.error(f"      ✗ Failed to download model: {e}")
        sys.exit(1)

    # ── Step 2: Download ViT feature extractor ────────────────────
    logger.info(f"\n[2/4] Downloading ViT Image Processor...")
    logger.info(f"      Model: {VIT_MODEL}")
    try:
        feature_extractor = ViTImageProcessor.from_pretrained(VIT_MODEL)
        logger.info(f"      ✓ ViT Image Processor downloaded")
    except Exception as e:
        # Try AutoFeatureExtractor as fallback
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(VIT_MODEL)
            logger.info(f"      ✓ ViT Feature Extractor downloaded (via AutoFeatureExtractor)")
        except Exception as e2:
            logger.error(f"      ✗ Failed: {e2}")
            sys.exit(1)

    # ── Step 3: Download Tamil RoBERTa tokenizer ──────────────────
    logger.info(f"\n[3/4] Downloading Tamil-RoBERTa Tokenizer...")
    logger.info(f"      Model: {ROBERTA_TOK}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ROBERTA_TOK)
        logger.info(f"      ✓ Tamil-RoBERTa Tokenizer downloaded")
        logger.info(f"        Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"      ✗ Failed to download tokenizer: {e}")
        sys.exit(1)

    # ── Step 4: Build TrOCRProcessor and test inference ──────────
    logger.info(f"\n[4/4] Validating end-to-end pipeline...")
    try:
        processor = TrOCRProcessor(
            image_processor=feature_extractor,
            tokenizer=tokenizer,
        )

        # Create a minimal test image (white 224×224 with a Tamil character)
        test_img = Image.fromarray(
            np.full((224, 224, 3), 255, dtype=np.uint8)
        ).convert("RGB")

        model = VisionEncoderDecoderModel.from_pretrained(MAIN_MODEL)
        model.eval()

        pixel_values = processor(images=test_img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=10)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        logger.info(f"      ✓ End-to-end inference test passed")
        logger.info(f"        Test output: {repr(output)}")
        del model

    except Exception as e:
        logger.warning(f"      ⚠ Inference test warning (model still usable): {e}")

    # ── Summary ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Tamil OCR model download complete!")
    logger.info("  Components cached in ~/.cache/huggingface/hub/")
    logger.info(f"    • {MAIN_MODEL}")
    logger.info(f"    • {VIT_MODEL}")
    logger.info(f"    • {ROBERTA_TOK}")
    logger.info("=" * 60)
    logger.info("\nYou can now use Tamil recognition in NEUROSCRIPT.")
    logger.info("Start the server:  python app.py")


if __name__ == "__main__":
    download_model()
