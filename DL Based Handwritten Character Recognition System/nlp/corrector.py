"""
nlp/corrector.py
────────────────
Language-aware NLP post-processing and spell correction.
All imports are lazy and guarded so the app works even without
optional packages installed.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Lazy singletons
_spell_en = None
_language_tool = None
_indic_normalizer_hi = None
_indic_normalizer_ta = None


def _get_spell_en():
    global _spell_en
    if _spell_en is None:
        try:
            from spellchecker import SpellChecker
            _spell_en = SpellChecker(language="en")
        except ImportError:
            logger.warning("pyspellchecker not installed — skipping English spell check")
            _spell_en = False
    return _spell_en


def _get_language_tool():
    global _language_tool
    if _language_tool is None:
        try:
            import language_tool_python
            _language_tool = language_tool_python.LanguageTool("en-US")
            logger.info("LanguageTool loaded")
        except Exception as e:
            logger.warning(f"LanguageTool not available: {e}")
            _language_tool = False
    return _language_tool


def _get_indic_normalizer(lang_code: str):
    global _indic_normalizer_hi, _indic_normalizer_ta
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        factory = IndicNormalizerFactory()
        if lang_code == "hi" and _indic_normalizer_hi is None:
            _indic_normalizer_hi = factory.get_normalizer("hi")
        if lang_code == "ta" and _indic_normalizer_ta is None:
            _indic_normalizer_ta = factory.get_normalizer("ta")
    except Exception as e:
        logger.debug(f"Indic NLP not available: {e}")
        return None
    return _indic_normalizer_hi if lang_code == "hi" else _indic_normalizer_ta


def _correct_english(text: str) -> tuple:
    corrections = []
    spell = _get_spell_en()

    if spell and spell is not False:
        words = text.split()
        corrected_words = []
        for word in words:
            stripped = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word)
            if len(stripped) < 2 or not stripped.isalpha():
                corrected_words.append(word)
                continue
            if stripped.lower() in spell:
                corrected_words.append(word)
            else:
                suggestion = spell.correction(stripped)
                if suggestion and suggestion.lower() != stripped.lower():
                    corrected = word.replace(stripped, suggestion)
                    corrections.append(f"'{stripped}' → '{suggestion}'")
                    corrected_words.append(corrected)
                else:
                    corrected_words.append(word)
        text = " ".join(corrected_words)

    tool = _get_language_tool()
    if tool and tool is not False:
        try:
            text = tool.correct(text)
        except Exception as e:
            logger.warning(f"Grammar tool error: {e}")

    return text, corrections


def _correct_indic(text: str, lang_code: str) -> tuple:
    normalizer = _get_indic_normalizer(lang_code)
    if normalizer is None:
        return text, []
    try:
        normalised = normalizer.normalize(text)
        notes = []
        if normalised != text:
            notes.append("Script normalised (Unicode canonical form applied)")
        return normalised, notes
    except Exception as e:
        logger.warning(f"Indic normalisation failed: {e}")
        return text, []


LANG_MAP = {"english": "en", "hindi": "hi", "tamil": "ta"}


def correct(text: str, lang: str) -> dict:
    """
    Apply NLP correction to OCR output.
    Returns dict: original, corrected, corrections, language
    """
    if not text or not text.strip():
        return {"original": text, "corrected": text, "corrections": [], "language": lang}

    lang_code = LANG_MAP.get(lang, "en")

    try:
        if lang_code == "en":
            corrected, corrections = _correct_english(text)
        elif lang_code in ("hi", "ta"):
            corrected, corrections = _correct_indic(text, lang_code)
        else:
            corrected, corrections = text, []
    except Exception as e:
        logger.warning(f"NLP correction failed: {e}")
        corrected, corrections = text, []

    logger.info(f"NLP ({lang}): {len(corrections)} correction(s)")
    return {"original": text, "corrected": corrected, "corrections": corrections, "language": lang}
