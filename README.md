# 设备健康评分

本地运行固定 ONNX 演示模型，对四项归一化传感器特征计算负荷评分。模型与样例均为合成数据，仅用于软件演示，不用于真实设备诊断。当前提供普通推理，不生成或验证零知识证明。

需要 Python 3.12。在仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m device_proof
.venv/bin/python -m device_proof model-check
.venv/bin/python scripts/demo_http.py --port 4311
.venv/bin/python -m uvicorn device_proof.api:app --host 127.0.0.1 --port 4311
.venv/bin/python -m pytest -q
```

`python -m device_proof model-check`（可选 `--model-id`，仅接受 `device-health-v1`）对固定模型执行发布契约完整性审计：清单须可解析且字段类型正确，`id`、版本、特征顺序、范围、输入输出形状、opset、IR 版本与 SHA-256 全部符合契约，并拒绝重复或未知特征；ONNX 工件须能解析并通过 checker，唯一输入 `features` 为 `FLOAT[1,4]`、唯一输出 `score` 为 `FLOAT[1,1]`，算子仅含 `MatMul` 与 `Add`，初始化器 `weights`、`bias` 的名称、类型、形状与常量值正确；随后在 CPU 上运行零值、内置示例与全一输入，结果须在 `1e-6` 内分别为 `0.1`、`0.4`、`1.0`。成功时在 stdout 输出单行 JSON 审计报告并以 `0` 退出；失败时以非零退出，stderr 仅含稳定错误码（如 `model_digest_mismatch`、`model_contract_violation`、`unknown_model`），不泄露路径、堆栈或工件内容。

网页入口为 `http://127.0.0.1:4311`。已有接口：`GET /healthz`、`GET /models`、`GET /models/{model_id}`、`GET /models/{model_id}/integrity`、`POST /score`；请求示例见 `examples/sample.json`，字段说明见 `/docs`。`GET /models/device-health-v1/integrity` 返回与 CLI 相同的审计报告；未知模型返回 `404`，工件缺失或未通过审计时返回 `503` 且 `detail` 为 `model_unavailable`。评分、会话创建与所有模型查询都必须先通过同一审计，审计失效即封锁；工件被进程内替换后缓存立即失效，绝不用旧会话继续提供结果。输入须为有限的 0–1 数值，模型仅支持 `device-health-v1`，仅提供普通推理，不生成或宣称零知识证明。可用 `python -m device_proof --input 文件路径` 计算其他输入。
