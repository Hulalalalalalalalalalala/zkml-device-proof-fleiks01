"""Rebuild the deterministic, synthetic ONNX demonstration model."""

from hashlib import sha256
import json
from pathlib import Path

from onnx import TensorProto, checker, helper, save


root = Path(__file__).resolve().parent.parent
output = root / "models"
output.mkdir(exist_ok=True)
model = helper.make_model(
    helper.make_graph(
        [
            helper.make_node("MatMul", ["features", "weights"], ["weighted"]),
            helper.make_node("Add", ["weighted", "bias"], ["score"]),
        ],
        "device-health-v1",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [1, 1])],
        [
            helper.make_tensor("weights", TensorProto.FLOAT, [4, 1], [0.25, 0.25, 0.25, 0.15]),
            helper.make_tensor("bias", TensorProto.FLOAT, [1], [0.1]),
        ],
    ),
    producer_name="device-health-synthetic-demo",
    opset_imports=[helper.make_opsetid("", 13)],
    ir_version=8,
)
checker.check_model(model)
path = output / "device-health-v1.onnx"
save(model, path)
manifest = {
    "id": "device-health-v1",
    "version": "1.0.0",
    "description": "Synthetic normalized device load score; not a diagnostic model.",
    "feature_order": ["temperature", "vibration", "current", "runtime"],
    "feature_range": [0.0, 1.0],
    "score_range": [0.1, 1.0],
    "score_direction": "higher-means-more-load",
    "input_shape": [1, 4],
    "output_shape": [1, 1],
    "onnx_opset": 13,
    "onnx_ir_version": 8,
    "sha256": sha256(path.read_bytes()).hexdigest(),
}
(output / "device-health-v1.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(path)
