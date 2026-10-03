# semantic-router-alp

Semantic Router 的云端 ALP 扩展，独立维护原生 Function Calling 适配、响应校验、私有历史配对和云端验证工具。

[English](README.md) · [协议行为](docs/protocol.md) · [配置](docs/configuration.md)

本项目与固定版本的 `semantic-router-multimodal` 补丁组合成同一个 SR 服务；部署由 `inference-stack` 管理。
它不依赖 vllm-alp、vLLM、MLX、XGrammar、PyTorch 或 GPU。

## 私有依赖

权威协议定义与校验来自私有仓库 `omni-runtime/alp_schema_mcp`。请自行取得访问权限，按
`dependencies.lock.json` 安装对应提交。这里不分发该依赖源码或完整契约。
包版本为 0.3.0，协议版本仍是 ALP 0.3.0 draft 1。

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '/authorized/path/alp_schema_mcp'
python -m pip install -e '.[test]'
python -m pytest
```

目录特化、宿主任务绑定和请求校验辅助代码由本插件内部维护，不需要单独的 `alp-core` 仓库、包或服务。

## 协议边界

六种操作及函数名来自权威 transport-map。`api_json` 只解码一次 payload_json；`typed` 使用
协议允许的静态投影。两者均恢复同一个 canonical 请求，再做完整协议、目录、动态参数与宿主约束校验。
每回合只接受一个完整动作，不补字段、不执行动作、不将供应商 strict 开关等同于完整协议保证。
SR 负责路由、云端凭据和私有历史；stdio worker 不发网络请求。实际授权、执行与持久 Run 属于宿主。

## 构建与测试

参见 [构建说明](docs/building.md) 和 [协议对应关系](docs/conformance.md)。
`patches/` 只维护 ALP 改动，准备脚本组合锁定的上游与多模态扩展。
`verify-alp.py` 使用外部 test-case，分别检查协议、任务匹配与原始调用映射。
验证报告只保存在本地，不纳入仓库；不会自动重试或把期望答案输入模型。

本仓库采用 Apache-2.0；私有依赖保留其各自条款。本项目不是 vLLM 官方发行版。
