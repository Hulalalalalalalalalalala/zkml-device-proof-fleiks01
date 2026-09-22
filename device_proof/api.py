from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import proofs
from .audit import AuditError, audit_model
from .scoring import MODEL_ID, ROOT, ScoreRequest, ScoreResponse, model_info, score, session
from .statements import QuantizationError, StatementResponse, build_statement


app = FastAPI(title="Device health scores", version="0.1.0")

# Restore persisted jobs (queued/running become failed/interrupted) and start
# the single FIFO proof worker.
proofs.recover_jobs()
proofs.start_worker()


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


@app.post("/proof-jobs", status_code=202)
def create_proof_job(request: ScoreRequest):
    try:
        return proofs.submit(request)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model") from None
    except QuantizationError as exc:
        raise HTTPException(status_code=422, detail=exc.code) from None
    except proofs.BackendUnavailable:
        raise HTTPException(status_code=503, detail=proofs.ERR_BACKEND_UNAVAILABLE) from None
    except AuditError:
        raise _unavailable() from None


@app.get("/proof-jobs/{job_id}")
def get_proof_job(job_id: str):
    job = proofs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown_proof_job")
    return job


@app.delete("/proof-jobs/{job_id}")
def delete_proof_job(job_id: str):
    result = proofs.cancel_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown_proof_job")
    if result is False:
        raise HTTPException(status_code=409, detail="proof_job_not_cancellable")
    return result


@app.post("/proof-verifications")
def create_proof_verification(payload=Body(default=None)):
    try:
        verified = proofs.verify_materials(payload)
    except proofs.MaterialError:
        raise HTTPException(status_code=422, detail=proofs.ERR_INVALID_MATERIAL) from None
    except proofs.BackendUnavailable:
        raise HTTPException(status_code=503, detail=proofs.ERR_BACKEND_UNAVAILABLE) from None
    except AuditError:
        raise _unavailable() from None
    return {"verified": verified}
