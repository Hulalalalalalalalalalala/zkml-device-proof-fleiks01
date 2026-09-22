# 设备健康评分

本地运行固定 ONNX 演示模型，对四项归一化传感器特征计算负荷评分。模型与样例均为合成数据，仅用于软件演示，不用于真实设备诊断。当前提供普通推理，不生成或验证零知识证明。

需要 Python 3.12。在仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m device_proof
.venv/bin/python scripts/demo_http.py --port 4311
.venv/bin/python -m uvicorn device_proof.api:app --host 127.0.0.1 --port 4311
.venv/bin/python -m pytest -q
```

网页入口为 `http://127.0.0.1:4311`。已有接口：`GET /healthz`、`GET /models`、`GET /models/{model_id}`、`GET /models/{model_id}/integrity`、`POST /score`、`POST /statements`；请求示例见 `examples/sample.json`，字段说明见 `/docs`。输入须为有限的 0–1 数值，模型仅支持 `device-health-v1`。可用 `python -m device_proof --input 文件路径` 计算其他输入。

评分、会话创建与模型查询前都会先执行固定模型完整性审计：校验清单可解析、字段类型正确且与发布契约一致（拒绝重复或未知特征），核对 ONNX 结构（唯一输入 `features` 为 FLOAT[1,4]、唯一输出 `score` 为 FLOAT[1,1]、仅含 MatMul 与 Add 算子、初始化器名称/类型/形状/常量值正确、opset 13、IR 8、SHA-256 匹配），并在 CPU 上以零值、现有示例与全一输入复算参考分数（1e-6 内分别为 0.1、0.4、1.0）。审计失败时服务进入失效封锁：相关接口返回 503（`model_unavailable`）。也可手动运行审计：

```bash
.venv/bin/python -m device_proof model-check [--model-id device-health-v1]
```

成功时输出单行 JSON 审计报告并以 0 退出；失败时退出码非零，stderr 仅含稳定错误码，不泄露路径、堆栈或工件内容。该审计仍属普通推理，不生成或宣称零知识证明。

## Q16 量化声明

`POST /statements` 与 CLI `statement --input <json>` 为同一份 `ScoreRequest` 出具固定点量化声明，量化版本为 `device-health-v1-q16-v1`（`scale=65536`、`rounding=ties-to-even`）。这仍属普通推理记账：不生成也不宣称零知识证明，不产出 witness 或 proof。

四项特征按清单 `feature_order`（temperature、vibration、current、runtime）依次精确编码为 q_t、q_v、q_c、q_r：对输入的二进制浮点值取 `Fraction` 后乘以 65536，按 roundTiesToEven 取整为 uint32；越界或溢出一律拒绝。编码值仅保存在进程内存的局部整数中，不记录、不落盘、不出现在任何响应里。

量化分数完全按有理数精确计算：

```
q_score = roundTiesToEven((q_t+q_v+q_c)/4 + 3*q_r/20 + 65536/10)
```

`q_score` 为 uint32；其值 `q_score/65536` 即负荷分数（越高负荷越高），与 `POST /score` 的误差不超过 `1/65536`。

成功响应（HTTP 与 CLI 为同一单行 JSON）含：`model_id`、`model_sha256`、`quantization_id`、`q_score`、`scale`、`rounding`、`statement_sha256`。声明不含原始特征、逐项编码值或 witness。`statement_sha256` 的原像是**除该字段外全部响应字段**组成的对象，按 RFC 8785（JCS）规范化为 UTF-8 字节后取 SHA-256，以 64 字符小写十六进制表示，跨进程一致。

```bash
.venv/bin/python -m device_proof statement --input examples/sample.json
```

失败时退出码非零且 stderr 仅含稳定错误码（如 `invalid_request`、`unknown_model`、`input_unreadable`、审计失败对应的码）；HTTP 沿用 422（输入或编码不合法）、404（未知模型），审计失败返回 503 `model_unavailable`。既有 `/score`、`model-check` 等功能保持兼容。
