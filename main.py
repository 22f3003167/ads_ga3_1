"""Multimodal QA API — accepts a base64 image + question, returns extracted answer.

POST /answer-image  {"image_base64": "...", "question": "..."}  ->  {"answer": "..."}
"""
import base64
import binascii
import os
import re

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Comma-separated fallback list. First model that succeeds wins.
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-2.5-flash,gemini-flash-latest,gemini-2.0-flash,gemini-2.0-flash-lite",
)
GEMINI_MODELS = [m.strip() for m in GEMINI_MODEL.split(",") if m.strip()]


def _model_url(model: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

SYSTEM_PROMPT = (
    "You are a precise document-extraction engine. You are given an image of a "
    "scanned document (a data table, bar chart, pie chart, receipt, or invoice) "
    "and a question about it. Read the image carefully and answer ONLY with the "
    "exact answer value — nothing else.\n"
    "Rules for the answer:\n"
    "- Return just the raw value. No explanation, no labels, no sentences.\n"
    "- For numeric answers, return ONLY the number (e.g. 4089.35). Do NOT include "
    "currency symbols (Rs, $, ₹), thousands separators, units, or the word 'total'.\n"
    "- Keep the decimal places exactly as they appear in the document when relevant.\n"
    "- If a calculation is required (e.g. a sum across bars), compute it and return "
    "only the resulting number.\n"
    "- If the answer is text (e.g. a vendor name), return the text exactly as shown.\n"
    "- If the value cannot be determined from the image, return the single word: unknown"
)

app = FastAPI(title="Multimodal Image QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageQARequest(BaseModel):
    image_base64: str
    question: str


def _clean_base64(raw: str) -> str:
    """Strip a data-URL prefix and whitespace from a base64 string."""
    raw = (raw or "").strip()
    if raw.startswith("data:"):
        # data:image/png;base64,XXXX
        comma = raw.find(",")
        if comma != -1:
            raw = raw[comma + 1:]
    return re.sub(r"\s+", "", raw)


def _detect_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _normalize_answer(text: str) -> str:
    """Trim model output down to the raw value."""
    ans = (text or "").strip()
    # Drop surrounding quotes/backticks the model sometimes adds.
    ans = ans.strip("`").strip().strip('"').strip("'").strip()
    # Take the first non-empty line (guards against stray trailing prose).
    for line in ans.splitlines():
        if line.strip():
            ans = line.strip()
            break
    return ans


def ask_gemini(image_b64: str, question: str) -> str:
    raw = _clean_base64(image_b64)
    try:
        img_bytes = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        img_bytes = b""
    mime = _detect_mime(img_bytes)

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": f"Question: {question}"},
                    {"inline_data": {"mime_type": mime, "data": raw}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
        },
    }

    last_err = None
    with httpx.Client(timeout=60) as client:
        for model in GEMINI_MODELS:
            try:
                resp = client.post(
                    _model_url(model),
                    params={"key": GEMINI_API_KEY},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                last_err = e
                body = e.response.text[:200] if e.response is not None else ""
                print(f"[{model}] {e.response.status_code}: {body}")
                continue
            try:
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts)
            except (KeyError, IndexError):
                text = ""
            if text.strip():
                return _normalize_answer(text)
    if last_err is not None:
        raise last_err
    return ""


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "POST /answer-image", "model": GEMINI_MODEL}


@app.get("/health")
def health():
    return {"status": "healthy", "gemini_key_set": bool(GEMINI_API_KEY)}


@app.post("/answer-image")
def answer_image(req: ImageQARequest):
    try:
        answer = ask_gemini(req.image_base64, req.question)
    except httpx.HTTPStatusError as e:
        answer = ""
        detail = e.response.text[:300] if e.response is not None else str(e)
        print(f"Gemini HTTP error: {detail}")
    except Exception as e:  # noqa: BLE001
        answer = ""
        print(f"Error: {e}")
    return {"answer": str(answer)}
