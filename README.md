# semantic-router-alp

Provider-native Agent Lifecycle Protocol (ALP) integration for Semantic Router.

[简体中文](README.zh-CN.md) · [Protocol behavior](docs/protocol.md) · [Configuration](docs/configuration.md)

This repository owns the SR native ALP transport extension, cloud Function Calling
projection, response validation adapter, private session pairing and cloud verifier.
It composes with the pinned `semantic-router-multimodal` extension into one SR image.
It is an independent integration project, not an official vLLM distribution.

## Dependency access

`omni-runtime/alp_schema_mcp` is a **private dependency**. Obtain access from its
maintainers before installing. Its source and contract bundle are not distributed
in this repository or public images. This project requires package version 0.3.1
and ALP protocol 0.3.0 draft 1. It does not depend on `vllm-alp`, vLLM, MLX,
XGrammar, PyTorch, a tokenizer or a GPU.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install '/authorized/path/alp_schema_mcp[runtime]'
python -m pip install -e '.[test]'
python -m pytest
```

Use the exact private dependency revision in `dependencies.lock.json`.
Configure SR to invoke `python -m semantic_router_alp.cloud_worker --config
/opt/alp/catalogs.json --projection typed`. See the configuration guide for a
complete command array and read-only worker mount. Deployment values and secrets
belong in `inference-stack`'s private instance configuration.

## Contracts and boundaries

The authoritative schema, operation map and native API envelope come from
`alp_schema_mcp`; there is no vendored protocol copy. The shared engine-independent
runtime specializes catalogs and trusted host requirements. This adapter projects
those contracts into native cloud tools and validates exactly one completed call.
The `api_json` projection decodes `payload_json` exactly once. The optional `typed`
projection restores the same canonical ALP object without repairing model output.
Provider strict mode is not treated as proof of complete protocol enforcement.

SR retains routing, provider credentials, dispatch and caller-bound private replay.
The worker uses bounded local stdio; it neither sends cloud requests nor executes
an agent or tool. Authorization and execution remain the host's responsibility.
Invalid, refused, truncated, mixed or parallel calls fail closed. Terminal calls
receive a paired local acknowledgement without another generation.

## Build and test

`dependencies.lock.json` pins the upstream and multimodal base. `patches/` contains
only the ALP extension. `scripts/prepare.py` composes and verifies both extensions;
no multimodal patch is copied into this repository. `scripts/test-native.sh`
runs the ALP Go tests in a prepared checkout with its native build dependencies.
See [building](docs/building.md) and [protocol requirements](docs/conformance.md).

Public CI checks source composition without private dependencies. Maintainers can
run the optional contract-test workflow with a separately configured, narrowly
scoped `ALP_CONTRACTS_TOKEN`. Native Go integration tests require the documented
build environment.

Run `scripts/verify-alp.py --help` for deployed cloud testing against an authorized
external producer suite. Protocol acceptance, task matching and exact native
mapping are separate checks. It never executes generated actions, inserts golden
answers or silently retries failed generations. Reports, credentials and deployment
inventories stay local and are excluded from Git.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/semantic_router_alp/` | Native projection and contract worker |
| `patches/` | SR ALP transport, configuration and replay integration |
| `scripts/` | Reproducible composition and deployed verification |
| `tests/` | Six-operation projection, failure and history checks |
| `examples/` | Synthetic catalogs; no deployment credentials |

[Apache-2.0](LICENSE). Private dependencies retain their own terms; this license
covers this repository only. Read [CONTRIBUTING.md](CONTRIBUTING.md) and
[SECURITY.md](SECURITY.md) before contributing or reporting vulnerabilities.
