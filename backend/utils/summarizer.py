import json
import logging
import os
from typing import Dict

from dotenv import load_dotenv
import re
import google.generativeai as genai

load_dotenv()

logger = logging.getLogger("meeting-summarizer.summarizer")

PROMPT = (
    "Summarize this meeting transcript into:\n"
    "1. Key decisions\n"
    "2. Action items (with responsible persons if mentioned)\n"
    "3. Overall summary"
)


def _fallback_summary(text: str) -> Dict[str, object]:
    snippet = (text or "").strip()
    if len(snippet) > 700:  # keep it concise
        snippet = snippet[:700] + "..."
    return {
        "key_decisions": [],
        "action_items": [],
        "overall_summary": snippet or "No summary available.",
    }


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

    seen = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            uniq.append(n)
            seen.add(n)
    return sorted(uniq, key=rank)

def summarize_transcript(transcript: str) -> Dict[str, object]:
    if not _configure_genai():
        return _fallback_summary(transcript)

    override = os.getenv("GEMINI_MODEL")
    # Prefer lower-cost flash models first to increase chances under free-tier or tight quotas
    base_list = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.5-pro",
    ]
    # Try dynamic discovery of available 2.x models
    dynamic = []
    try:
        models = list(genai.list_models())
        for m in models:
            name = getattr(m, 'name', '')
            if name.startswith('models/'):
                name = name.split('/', 1)[1]
            methods = set(getattr(m, 'supported_generation_methods', []) or [])
            if name.startswith('gemini-2') and ('generateContent' in methods or 'generate_content' in methods):
                dynamic.append(name)
    except Exception:
        pass
    ordered_dynamic = _prefer_model_order(dynamic)
    model_candidates = ordered_dynamic or [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.0-pro",
        "gemini-2.5-pro",
    ]
    model_candidates = _prefer_model_order(model_candidates)
    # Reduce input size to avoid hitting input token quotas; configurable via SUMMARY_MAX_CHARS
    try:
        max_chars = int(os.getenv("SUMMARY_MAX_CHARS", "12000"))
    except Exception:
        max_chars = 12000
    t = transcript if len(transcript or "") <= max_chars else (transcript[:max_chars] + "\n[...truncated]")

    prompt = (
        f"{PROMPT}\n\nTranscript:\n{t}\n\n"
        "Respond ONLY with valid JSON using exactly these keys and constraints: "
        "{\"key_decisions\": string[], \"action_items\": string[], \"overall_summary\": string}. "
        "- key_decisions: up to 5 concise bullets (no duplicates). "
        "- action_items: up to 8 concise bullets, include responsible person if present. "
        "- overall_summary: 2-5 sentences. "
        "Return raw JSON only. Do not include markdown, comments, or explanations."
    )
    last_err = None
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            content = (resp.text or "").strip()

            # Strip code fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)

            # Try direct parse first
            try:
                data = json.loads(content)
            except Exception:
                # Fallback: extract first JSON object substring
                match = re.search(r"\{[\s\S]*\}", content)
                if not match:
                    logger.warning("No JSON object found in model output. Raw: %s", content)
                    return _fallback_summary(transcript)
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    logger.warning("Failed to parse extracted JSON. Raw: %s", match.group(0))
                    return _fallback_summary(transcript)

            # Normalize and cap lengths
            kd = [s for s in (data.get("key_decisions", []) or []) if isinstance(s, str) and s.strip()]
            ai = [s for s in (data.get("action_items", []) or []) if isinstance(s, str) and s.strip()]
            osum = data.get("overall_summary", "") or ""
            return {
                "key_decisions": kd[:5],
                "action_items": ai[:8],
                "overall_summary": osum.strip(),
            }
        except Exception as e:
            last_err = e
            logger.warning("Model %s failed: %s", model_name, e)
            continue
    if last_err:
        logger.exception("Gemini summarization error: %s", last_err)
    return _fallback_summary(transcript)
