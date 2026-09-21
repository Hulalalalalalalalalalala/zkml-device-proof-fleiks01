from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .integrity import MODEL_ID, UNKNOWN_MODEL, IntegrityError
from .scoring import ROOT, ScoreRequest, ScoreResponse, guard, model_info, score


app = FastAPI(title="Device health scores", version="0.1.0")


def _raise_for_integrity(error: IntegrityError) -> None:
    """Map a stable audit code onto the HTTP contract. No paths or internals."""
    if error.code == UNKNOWN_MODEL:
        raise HTTPException(status_code=404, detail="Unknown model") from None
    raise HTTPException(status_code=503, detail="model_unavailable") from None


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def health():
    try:
        guard.access()
    except IntegrityError as error:
        _raise_for_integrity(error)
    return {"status": "ok", "engine": "onnxruntime", "mode": "ordinary-inference"}


@app.get("/models")
def models():
    try:
        return {"items": [model_info()]}
    except IntegrityError as error:
        _raise_for_integrity(error)


@app.get("/models/{model_id}/integrity")
def model_integrity(model_id: str):
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail="Unknown model")
    try:
        return guard.access().report
    except IntegrityError as error:
        _raise_for_integrity(error)


@app.get("/models/{model_id}")
def get_model(model_id: str):
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail="Unknown model")
    try:
        return model_info()
    except IntegrityError as error:
        _raise_for_integrity(error)


@app.post("/score", response_model=ScoreResponse)
def create_score(request: ScoreRequest):
    try:
        return score(request)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model") from None
    except IntegrityError as error:
        _raise_for_integrity(error)
