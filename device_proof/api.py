from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .scoring import MODEL_ID, ROOT, ScoreRequest, ScoreResponse, model_info, score, session


app = FastAPI(title="Device health scores", version="0.1.0")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def health():
    session()
    return {"status": "ok", "engine": "onnxruntime", "mode": "ordinary-inference"}


@app.get("/models")
def models():
    return {"items": [model_info()]}


@app.get("/models/{model_id}")
def get_model(model_id: str):
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail="Unknown model")
    return model_info()


@app.post("/score", response_model=ScoreResponse)
def create_score(request: ScoreRequest):
    try:
        return score(request)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model") from None
