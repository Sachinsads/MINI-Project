"""
download_models.py
──────────────────
One-time script to pre-download all three HuggingFace OCR models
to the local cache (~/.cache/huggingface/hub/) so that the Flask
server can load them instantly without internet access at runtime.

Usage:
    python download_models.py
"""

from huggingface_hub import snapshot_download
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
)
import os

MODELS = {
    "english": {
        "name": "Microsoft TrOCR (English Handwritten)",
        "repo": "microsoft/trocr-base-handwritten",
        "type": "trocr",
    },
    "hindi": {
        "name": "Hindi Offline Handwritten OCR",
        "repo": "sabaridsnfuji/Hindi_Offline_Handwritten_OCR",
        "type": "ved",
        "encoder": "google/vit-base-patch16-224-in21k",
        "decoder": "surajp/RoBERTa-hindi",
    },
    "tamil": {
        "name": "Tamil Offline Handwritten OCR",
        "repo": "sabaridsnfuji/Tamil_Offline_Handwritten_OCR",
        "type": "ved",
        "encoder": "google/vit-base-patch16-224-in21k",
        "decoder": "d42kw01f/Tamil-RoBERTa",
    },
}


def download_model(lang, info):
    print(f"\n{'='*60}")
    print(f"  Downloading: {info['name']}")
    print(f"  HuggingFace ID: {info['repo']}")
    print(f"{'='*60}")

    try:
        if info["type"] == "trocr":
            print("  → Downloading processor …")
            TrOCRProcessor.from_pretrained(info["repo"])
            print("  → Downloading model weights …")
            VisionEncoderDecoderModel.from_pretrained(info["repo"])
        else:
            # Vision Encoder-Decoder: download main model repo
            print("  → Downloading model weights …")
            VisionEncoderDecoderModel.from_pretrained(info["repo"])

            # Download encoder feature extractor
            print(f"  → Downloading encoder ({info['encoder']}) …")
            AutoFeatureExtractor.from_pretrained(info["encoder"])

            # Download decoder tokenizer
            print(f"  → Downloading decoder ({info['decoder']}) …")
            AutoTokenizer.from_pretrained(info["decoder"])

        print(f"  ✓  {info['name']} — downloaded successfully!")
        return True

    except Exception as e:
        print(f"  ✗  Failed to download {info['name']}: {e}")
        print(f"     The model will be downloaded on first use instead.")
        return False


def main():
    print("\n" + "="*60)
    print("  Handwritten OCR — Model Downloader")
    print("  This may take several minutes (total ~2 GB).")
    print("="*60)

    results = {}
    for lang, info in MODELS.items():
        results[lang] = download_model(lang, info)

    print("\n" + "="*60)
    print("  Download Summary")
    print("="*60)
    for lang, success in results.items():
        status = "✓ Downloaded" if success else "✗ Failed (will download on first use)"
        print(f"  {lang.capitalize():8s} : {status}")
    print("="*60)
    print("\nDone! You can now start the server with:  python app.py\n")


if __name__ == "__main__":
    main()
