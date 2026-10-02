"""
utils/image_processor.py
────────────────────────
High-precision image preprocessing & line segmentation pipeline for handwritten text.

Key feature: Binarization & projection profiles are used ONLY to identify text line
bounding boxes. The actual line images passed to TrOCR are cropped from the high-quality
RGB image (with CLAHE contrast enhancement) to preserve smooth anti-aliased handwriting strokes.
"""

import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def enhance_contrast_rgb(pil_img: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Convert PIL Image to OpenCV BGR, apply LAB CLAHE contrast enhancement,
    and return (enhanced_pil, gray_array).
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Convert to LAB space for luminance enhancement without distorting color
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(enhanced_rgb), gray


def binarize_for_segmentation(gray: np.ndarray) -> np.ndarray:
    """Apply bilateral filtering + Otsu's thresholding to get binary mask for line detection."""
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if text is dark on light background (so text pixels = 255)
    if np.mean(binary) > 127:
        inv = cv2.bitwise_not(binary)
    else:
        inv = binary
    return inv


def get_line_bounding_boxes(inv_binary: np.ndarray, min_height: int = 15) -> list[tuple[int, int]]:
    """
    Find vertical start and end indices of text lines using horizontal projection profile.
    """
    proj = np.sum(inv_binary, axis=1)
    # Threshold out noise
    max_proj = np.max(proj) if np.max(proj) > 0 else 1
    row_mask = proj > (max_proj * 0.02)

    boxes = []
    in_line = False
    start = 0

    for i, active in enumerate(row_mask):
        if not in_line and active:
            in_line = True
            start = i
        elif in_line and not active:
            in_line = False
            if (i - start) >= min_height:
                boxes.append((start, i))

    if in_line and (len(row_mask) - start) >= min_height:
        boxes.append((start, len(row_mask)))

    return boxes


def preprocess_and_segment(pil_image: Image.Image) -> tuple[list[Image.Image], np.ndarray]:
    """
    Full preprocessing pipeline.

    Args:
        pil_image: Input PIL Image

    Returns:
        (line_pil_images, preprocessed_binary) where:
          - line_pil_images: list of PIL RGB images (one per text line)
          - preprocessed_binary: binarized numpy array (for UI display tab)
    """
    enhanced_pil, gray = enhance_contrast_rgb(pil_image)
    inv_binary = binarize_for_segmentation(gray)

    boxes = get_line_bounding_boxes(inv_binary)
    h, w = gray.shape

    line_pil_images = []
    pad_y = 6
    pad_x = 10

    if not boxes or len(boxes) == 0:
        # Fallback: treat whole image as single line
        boxes = [(0, h)]

    for (top, bottom) in boxes:
        y1 = max(0, top - pad_y)
        y2 = min(h, bottom + pad_y)

        # Crop line from the high-quality enhanced RGB image
        line_crop = enhanced_pil.crop((0, y1, w, y2))

        # Crop horizontal whitespace
        line_inv = inv_binary[y1:y2, :]
        col_proj = np.sum(line_inv, axis=0)
        nonzero = np.where(col_proj > 0)[0]

        if len(nonzero) > 0:
            x1 = max(0, nonzero[0] - pad_x)
            x2 = min(w, nonzero[-1] + pad_x)
            line_crop = enhanced_pil.crop((x1, y1, x2, y2))

        # Resize for TrOCR if height is below 38px
        lw, lh = line_crop.size
        if lh < 38:
            scale = 38.0 / max(1, lh)
            new_w = max(38, int(lw * scale))
            line_crop = line_crop.resize((new_w, 38), Image.LANCZOS)

        line_pil_images.append(line_crop)

    display_binary = cv2.bitwise_not(inv_binary)
    logger.info(f"Preprocessed image into {len(line_pil_images)} text line(s)")
    return line_pil_images, display_binary
