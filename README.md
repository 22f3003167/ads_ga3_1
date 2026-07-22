# Multimodal Image QA API

`POST /answer-image` → accepts a base64 image + a question, returns `{"answer": "..."}`.
Uses Google Gemini (vision). CORS is open (`*`) so the grader's Cloudflare Worker can call it.

## Endpoints
- `POST /answer-image` — body `{"image_base64": "...", "question": "..."}` → `{"answer": "..."}`
- `GET /` and `GET /health` — status checks

## Local run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key"
uvicorn main:app --host 0.0.0.0 --port 8000
```
Test:
```bash
python3 - <<'PY'
import base64, requests
b = base64.b64encode(open("test_invoice.png","rb").read()).decode()
r = requests.post("http://localhost:8000/answer-image",
                  json={"image_base64": b, "question": "What is the grand total?"})
print(r.json())
PY
```

## Deploy to Render
1. Push this folder to a GitHub repo.
2. Render → **New → Web Service** → connect the repo.
3. Render auto-detects `render.yaml`. Settings (if manual):
   - Runtime: **Python**
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Plan: **Free**
4. In **Environment**, add:
   - `GEMINI_API_KEY` = your key
   - `GEMINI_MODEL` = `gemini-2.0-flash` (optional; comma-separated list allowed for fallback)
5. Deploy. Your base URL will be `https://<name>.onrender.com`.
6. Submit `https://<name>.onrender.com` (the grader hits `/answer-image`).

## Notes
- The `answer` is always a string. The prompt instructs the model to return the raw value only (no currency/units).
- `GEMINI_MODEL` accepts a comma-separated fallback list; the first model that responds wins.
- Free Render instances cold-start (~30–50s) after idle; the first grader request may be slow.
