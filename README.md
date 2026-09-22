# 设备健康评分

本地运行固定 ONNX 演示模型，对四项归一化传感器特征计算负荷评分。模型与样例均为合成数据，仅用于软件演示，不用于真实设备诊断。普通推理由 `POST /score` 提供；此外 `POST /proof-jobs` 在本地 CPU 上调用真实的 EZKL 23.0.5 为量化声明生成零知识证明，`POST /proof-verifications` 用 EZKL 校验证明材料（不使用任何模拟或占位）。

需要 Python 3.12。在仓库根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m device_proof
.venv/bin/python scripts/demo_http.py --port 4311
.venv/bin/python -m uvicorn device_proof.api:app --host 127.0.0.1 --port 4311
.venv/bin/python -m pytest -q
```

网页入口为 `http://127.0.0.1:4311`。已有接口：`GET /healthz`、`GET /models`、`GET /models/{model_id}`、`GET /models/{model_id}/integrity`、`POST /score`、`POST /statements`、`POST /proof-jobs`、`GET /proof-jobs/{job_id}`、`DELETE /proof-jobs/{job_id}`、`POST /proof-verifications`；请求示例见 `examples/sample.json`，字段说明见 `/docs`。输入须为有限的 0–1 数值，模型仅支持 `device-health-v1`。可用 `python -m device_proof --input 文件路径` 计算其他输入。

评分、会话创建、量化声明与证明任务在执行前都会先运行固定模型完整性审计：校验清单可解析、字段类型正确且与发布契约一致（拒绝重复或未知特征），核对 ONNX 结构（唯一输入 `features` 为 FLOAT[1,4]、唯一输出 `score` 为 FLOAT[1,1]、仅含 MatMul 与 Add 算子、初始化器名称/类型/形状/常量值正确、opset 13、IR 8、SHA-256 匹配），并在 CPU 上以零值、现有示例与全一输入复算参考分数（1e-6 内分别为 0.1、0.4、1.0）。审计失败时服务进入失效封锁：相关接口返回 503（普通推理为 `model_unavailable`）。也可手动运行审计：

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

## 本地 CPU 异步证明任务

`POST /proof-jobs` 复用与 `/score`、`/statements` 完全相同的 `ScoreRequest`。服务先运行模型完整性审计、计算 Q16 量化声明，再确认本地 EZKL 后端可用，随后把任务放入**单工作器 FIFO 队列**并立即返回 `202 Accepted`：

```json
{
  "job_id": "…32 位十六进制…",
  "status": "queued",
  "queued": true,
  "model_id": "device-health-v1",
  "model_sha256": "…",
  "quantization_id": "device-health-v1-q16-v1",
  "q_score": 26214,
  "scale": 65536,
  "rounding": "ties-to-even",
  "statement_sha256": "…",
  "ezkl_version": "23.0.5"
}
```

状态机只有 `queued`、`running`、`succeeded`、`failed`、`cancelled` 五种，终态不可逆：`queued → running → succeeded/failed`，`queued → cancelled`。`GET /proof-jobs/{job_id}` 查询状态；`DELETE` 只能取消仍在 `queued` 的任务（成功返回该任务视图），未知任务返回 404 `proof_job_not_found`，`running` 或任何终态返回 409 `proof_job_not_cancellable`。证明在本地 CPU 上由真实 EZKL 23.0.5 完成（`gen_settings → calibrate_settings → compile_circuit → get_srs → setup → gen_witness → prove`，输入可见性 `Private`、输出 `Public`），运行期任何失败都记为稳定错误码 `proof_generation_failed`，内部细节不进入响应、日志或异常文本。

### 隐私与持久化边界

只持久化**非敏感元数据、稳定错误码和公开材料**（位于运行目录 `runtime/`）：任务记录含声明字段、状态与时间戳；成功任务另存五件公开材料。原始特征、逐项 Q16 编码与 witness **只驻留内存**：证明期间它们仅写入每个任务私有的临时目录（优先 tmpfs `/dev/shm`），任务一到终态该目录立即整体删除；它们永不落任务库、永不进入响应、日志或异常。重启时，上一进程的 `queued`/`running` 任务因私有输入只存在于旧进程内存而无法续作，一律置为 `failed`（错误码 `interrupted`），同一请求可重新提交；`succeeded` 任务的公开材料仍可读取与复验。无 EZKL 后端时，提交与验证接口返回 503 `proof_backend_unavailable`。

### 证明材料与 manifest

成功任务的 `GET` 视图带 `materials`，其中 `proof`、`verification_key`、`settings` 为 Base64，`manifest` 为对象，`instances` 为公开实例数组（仅含一个公开输出 felt，**不含任何私有输入**）。电路以 Q16（`input/param/output scale = 2^16`）构建，公开输出 felt 与声明 `q_score` 位于同一定点网格，EZKL 再量化误差不超过 1 个定点单位（manifest 中 `public_output.felt` 与 `q_score` 之差 ≤ 1）。

manifest 用 SHA-256 把声明与全部材料绑定：含 `model_id`、`model_sha256`、`quantization_id`、`q_score`、`scale`、`rounding`、`statement_sha256`、`ezkl_version`、电路可见性/刻度、`public_output`、`instances_sha256`，以及 `proof_sha256`、`verification_key_sha256`、`settings_sha256`。验证密钥对同一 ONNX/电路确定性生成（跨重建摘要一致），是“证明究竟跑在哪个模型/电路上”的锚点。

### 证明验证

`POST /proof-verifications` 接收 `{manifest, proof, verification_key, settings}`（均为字符串：manifest 为 JSON 文本，其余为 Base64）。服务依次：校验结构与类型 → 复核 proof/settings/vk/instances 各摘要与 manifest 一致 → 复核输出 felt 与 `q_score` 一致（≤1）且内嵌声明摘要正确 → 校验量化方案被钉死为 `device-health-v1-q16-v1`（防止把真实证明重标记为别的方案）→ 校验模型映射（`model_id`/`model_sha256` 与受信发布模型一致，vk/settings 与受信电路一致）→ 最后调用真实 `ezkl.verify`。成功返回 `{verified: true, model_id, model_sha256, quantization_id, q_score, statement_sha256, instances, ezkl_version}`；任何结构、摘要、模型映射或密码学校验失败返回 422 `invalid_proof_material`；后端缺失返回 503 `proof_backend_unavailable`。

首次构建电路会生成并缓存确定性公共 KZG SRS、编译电路与密钥（缓存于运行目录并以一份真实自检测试证明校验，之后跨进程秒级复用）；所有既有接口行为保持不变。

## 版本化证据包

CLI `export-bundle` 把某个 **succeeded** 证明任务的全部公开材料打包为单个 zip 证据包：

```bash
.venv/bin/python -m device_proof export-bundle --job-id <id> --output <zip> [--runtime-dir runtime]
.venv/bin/python -m device_proof verify-bundle --bundle <zip> [--backend-dir runtime/backend]
```

zip 根目录**仅含**六个固定成员：`bundle.json`、`manifest.json`、`proof.json`、`verification_key.key`、`settings.json`、`instances.json`。`bundle.json` 含 `bundle_version=1` 及其余五个文件的 SHA-256；包内不含原始特征、逐项编码、witness 或任何内部路径，成员为固定相对名、固定时间戳的普通文件。

`verify-bundle` 完全独立校验：**不启动 HTTP、不读取 runtime 任务库**。它先限制成员数与总展开大小，拒绝绝对路径、`..`、重复名、目录、链接、加密成员与压缩炸弹；再依次检查 `bundle_version` 与成员集合、`bundle.json` 摘要、manifest 与各材料及 `instances.json` 的摘要绑定、受信模型与量化映射（`model_id`/`model_sha256`/vk/settings 与受信发布一致、量化钉死为 `device-health-v1-q16-v1`），最后调用真实 `ezkl.verify`。成功输出与 `POST /proof-verifications` 相同的单行 JSON（`verified: true` 等）并以 0 退出。

失败时退出码非零且 stderr 仅含稳定错误码：未知任务 `proof_job_not_found`、任务未成功 `proof_bundle_not_ready`、未知包版本 `unsupported_bundle_version`、模型或量化冲突 `untrusted_model`、包结构或安全问题 `invalid_bundle`、材料或验真失败 `invalid_proof_material`、后端缺失 `proof_backend_unavailable`。既有接口与 CLI 行为保持兼容。
