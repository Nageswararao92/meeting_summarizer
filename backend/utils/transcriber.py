import logging
import os
import time
from typing import Optional
import base64
import mimetypes

from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import speech_v1p1beta1 as speech

load_dotenv()

logger = logging.getLogger("meeting-summarizer.transcriber")


def _configure_genai() -> bool:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not set in environment.")
        return False
    genai.configure(api_key=api_key)
    return True


def _prefer_model_order(names):
    """Sort models by preference: flash-lite, flash, pro(non-preview), preview/exp, others."""
    def rank(n: str):
        s = (n or "").lower()
        if "flash-lite" in s:
            return (0, len(s))
        if "flash" in s:
            return (1, len(s))
        if "-pro" in s and "preview" not in s and "-exp" not in s:
            return (2, len(s))
        if "preview" in s or "-exp" in s:
            return (3, len(s))
        return (4, len(s))

    # de-dupe preserving first occurrence, then sort by rank
    seen = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            uniq.append(n)
            seen.add(n)
    return sorted(uniq, key=rank)


def _transcribe_with_gemini(file_path: str, max_retries: int, backoff: float, timeout: int) -> Optional[str]:
    if not _configure_genai():
        return None
    override = os.getenv("GEMINI_MODEL")
    # Try dynamic discovery of available 2.x models
    dynamic = []
    try:
        models = list(genai.list_models())
        for m in models:
            name = getattr(m, 'name', '')
            # strip leading 'models/' if present
            if name.startswith('models/'):
                name = name.split('/', 1)[1]
            methods = set(getattr(m, 'supported_generation_methods', []) or [])
            if name.startswith('gemini-2') and ('generateContent' in methods or 'generate_content' in methods):
                dynamic.append(name)
    except Exception:
        pass

    # Prefer lower-cost flash models first to increase chances under free-tier or tight quotas
    ordered_dynamic = _prefer_model_order(dynamic)
    base_list = ordered_dynamic or [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.5-pro",
    ]
    model_candidates = ([override] + base_list) if override else base_list
    model_candidates = _prefer_model_order(model_candidates)
    try:
        logger.info("Gemini ASR model order: %s", ", ".join(model_candidates))
    except Exception:
        pass
    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            mime, _ = mimetypes.guess_type(file_path)
            if not mime:
                ext = os.path.splitext(file_path)[1].lower()
                mime = {
                    ".mp3": "audio/mpeg",
                    ".wav": "audio/wav",
                    ".m4a": "audio/mp4",
                    ".aac": "audio/aac",
                    ".ogg": "audio/ogg",
                    ".flac": "audio/flac",
                }.get(ext, "audio/mpeg")
            audio_part = {"mime_type": mime, "data": audio_bytes}
            prompt = "Transcribe the audio content to plain text. Return only the transcript text."
            last_err = None
            for model_name in model_candidates:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content([audio_part, prompt])
                    text = (resp.text or "").strip()
                    if text:
                        return text
                except Exception as me:
                    last_err = me
                    logger.warning("Model %s failed: %s", model_name, me)
                    continue
            if last_err:
                raise last_err
            return None
        except Exception as e:
            logger.warning("Gemini transcription attempt %d/%d failed: %s", attempt, max_retries, e)
            if attempt == max_retries:
                logger.exception("Transcription failed after %d attempts", max_retries)
                return None
            time.sleep(backoff ** attempt)
    return None


def _transcribe_with_google_speech(file_path: str, *, max_retries: int, backoff: float) -> Optional[str]:
    # Requires GOOGLE_APPLICATION_CREDENTIALS or credentials set up in environment
    client = speech.SpeechClient()
    with open(file_path, "rb") as f:
        content = f.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        enable_automatic_punctuation=True,
        language_code=os.getenv("ASR_LANGUAGE", "en-US"),
        model=os.getenv("ASR_GOOGLE_MODEL", "latest_long"),
        audio_channel_count=2,
        enable_separate_recognition_per_channel=False,
    )
    try:
        if len(content) > 10 * 1024 * 1024:  # >10MB, use long running
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result(timeout=600)
        else:
            response = client.recognize(config=config, audio=audio)
        parts = []
        for result in response.results:
            alt = result.alternatives[0]
            parts.append(alt.transcript)
        text = " ".join(parts).strip()
        return text or None
    except Exception as e:
        logger.exception("Google Speech transcription error: %s", e)
        return None


def transcribe_audio(file_path: str, *, max_retries: int = 3, backoff: float = 1.5, timeout: int = 300) -> Optional[str]:
    provider = (os.getenv("ASR_PROVIDER", "gemini") or "gemini").lower()
    if provider == "google_speech":
        return _transcribe_with_google_speech(file_path, max_retries=max_retries, backoff=backoff)
    # default gemini
    return _transcribe_with_gemini(file_path, max_retries=max_retries, backoff=backoff, timeout=timeout)
