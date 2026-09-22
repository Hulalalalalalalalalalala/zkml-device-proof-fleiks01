from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .audit import AuditError, audit_model
from .scoring import MODEL_ID, ROOT, ScoreRequest, ScoreResponse, model_info, score, session
from .statements import QuantizationError, StatementResponse, build_statement


app = FastAPI(title="Device health scores", version="0.1.0")


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="model_unavailable")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def health():
    try:
        session()
    except AuditError:
        raise _unavailable() from None
    return {"status": "ok", "engine": "onnxruntime", "mode": "ordinary-inference"}


@app.get("/models")
def models():
    try:
        return {"items": [model_info()]}
    except AuditError:
        raise _unavailable() from None


@app.get("/models/{model_id}")
def get_model(model_id: str):
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail="Unknown model")
    try:
        return model_info()
    except AuditError:
        raise _unavailable() from None


@app.get("/models/{model_id}/integrity")
def model_integrity(model_id: str):
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail="Unknown model")
    try:
        return audit_model(model_id)
    except AuditError:
        raise _unavailable() from None


@app.post("/score", response_model=ScoreResponse)
def create_score(request: ScoreRequest):
    try:
        return score(request)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model") from None
    except AuditError:
        raise _unavailable() from None


@app.post("/statements", response_model=StatementResponse)
def create_statement(request: ScoreRequest):
    try:
        return build_statement(request)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model") from None
    except QuantizationError as exc:
        raise HTTPException(status_code=422, detail=exc.code) from None
    except AuditError:
        raise _unavailable() from None
