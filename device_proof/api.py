from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from .audit import AuditError, audit_model
from .backend import ProofBackendError
from .jobs import JobError, get_manager
from .proof_materials import MaterialError
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


# -- asynchronous local CPU proof jobs -------------------------------------

_CLAIM_FIELDS = (
    "model_id", "model_sha256", "quantization_id", "q_score", "scale",
    "rounding", "statement_sha256", "ezkl_version",
)


def _claim(record: dict) -> dict:
    return {key: record[key] for key in _CLAIM_FIELDS}


def _job_view(record: dict) -> dict:
    """Public job view. Never includes features, encodings, or witnesses."""
    view = {
        "job_id": record["job_id"],
        "status": record["status"],
        "error_code": record.get("error_code"),
        "created_at": record.get("created_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        **_claim(record),
    }
    if "materials" in record:
        view["materials"] = record["materials"]
    return view


class ProofVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: str
    proof: str
    verification_key: str
    settings: str


@app.post("/proof-jobs", status_code=202)
def create_proof_job(request: ScoreRequest):
    manager = get_manager()
    try:
        record = manager.submit(request)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown model") from None
    except QuantizationError as exc:
        raise HTTPException(status_code=422, detail=exc.code) from None
    except AuditError:
        raise _unavailable() from None
    except ProofBackendError as exc:
        raise HTTPException(status_code=503, detail=exc.code) from None
    return {
        "job_id": record["job_id"],
        "status": record["status"],
        "queued": record["status"] == "queued",
        **_claim(record),
    }


@app.get("/proof-jobs/{job_id}")
def get_proof_job(job_id: str):
    record = get_manager().get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="proof_job_not_found")
    return _job_view(record)


@app.delete("/proof-jobs/{job_id}")
def cancel_proof_job(job_id: str):
    try:
        record = get_manager().cancel(job_id)
    except JobError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.code) from None
    return _job_view(record)


@app.post("/proof-verifications")
def verify_proof(request: ProofVerificationRequest):
    manager = get_manager()
    try:
        return manager.verify_payload(request.model_dump())
    except MaterialError:
        raise HTTPException(status_code=422, detail="invalid_proof_material") from None
    except ProofBackendError as exc:
        raise HTTPException(status_code=503, detail=exc.code) from None
