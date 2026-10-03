# Cloud native contract worker

SR can call `python -m semantic_router_alp.cloud_worker --config catalogs.json` as a bounded
stdio process. SR owns provider routing, HTTP requests, credentials and private
replay storage. This worker compiles ALP contracts and validates one generated
native Function Call; it makes no network request and executes no agent or tool.
It uses the common protocol code without loading a tokenizer, GPU or vLLM engine.

The authoritative `alp_schema_mcp` transport map supplies all six function names.
Two operator-selected projections are supported:

| Projection | Native function arguments | Provider strict request |
| --- | --- | --- |
| `api_json` (default) | `protocol_version`, `request_id`, `payload_json` string | Requested for the outer envelope |
| `typed` | `protocol_version`, `request_id`, `payload` object | Off unless `--request-strict` is supplied |

The protocol permits operation-specific static native projections. Typed mode
compiles the operation, catalog and host contracts into native parameters. It
adds only the operation identified by the authoritative function-name mapping
when constructing canonical requests. It never fills or rewrites a generated
payload. Implied types and closed-union properties are exposed for native tool
consumers while retaining all original references and union assertions. Literal
host values, including capability JSON Schemas, remain exact.


Signed or catalog-bound tasks retain the smaller native schema view: fixed host
values and original union assertions remain intact, without duplicating union
properties. Unbound tasks expose the redundant type/property information.
`alp.native_schema_view` reports `bound_minimal`, `expanded` or `api_envelope`.
This affects schema presentation only; the canonical and host validators are
identical and no provider output is filled or repaired.

## SR integration

Install authorized copies of this package and `alp_schema_mcp` in the worker
environment, then configure SR's `global.integrations.alp.command`:

```yaml
command:
  - /opt/alp/bin/python
  - -m
  - semantic_router_alp.cloud_worker
  - --config
  - /etc/alp/catalogs.json
  - --projection
  - typed
worker_env: [ALP_HOST_TASK_KEY]
```

Only the operator selects a projection, command or environment allowlist. Do not
pass cloud API credentials to the worker. The alp_schema_mcp dependency remains private; obtain access separately.
SR configuration, replay and deployment details are documented in
[SR native ALP](protocol.md).

Signed `AgentCallTask` and `DefinitionTask` inputs use the same constraints as
local engines; see [task constraints](protocol.md#request-and-validation). Typed mode places
these requirements directly in tool parameters and omits the duplicate action
schema from the system context. API JSON mode describes bound payload schemas
on the `payload_json` string. Neither mode guarantees that a cloud model follows
every schema assertion. Unlike local token masks, these are provider requests.

## Validation and replay contract

The worker accepts one complete native function call in 0.3, or an atomic collection
of 1–16 ordinary calls in opt-in 0.4. It rejects unknown
operations, extra outer fields, duplicate JSON keys, nonfinite numbers,
truncation, refusal, parallel calls in 0.3 and mixed explanatory content. See
[0.4 result pairing and exclusive calls](protocol-04.md). Whitespace-only
assistant content is framing; it is retained in private history and reported in
`transport_whitespace_ignored`. Full canonical ALP, dynamic catalog and signed
host constraints are checked after generation. The generation schema is also
validated in typed mode. External schema retrieval is disabled.

`strict: true` is never treated as proof of provider enforcement. Responses expose
`provider_strict_enforcement: not_requested` or `not_verified`,
`provider_projection`, `validation_scope` and `task_constraint_coverage`.
`validated: true` still means neither execution nor authorization. The adapter
performs no response repair, fallback model selection or hidden model retry.

With `include_raw: true`, `raw` contains the original native function name and
argument string. `raw_codec` is `provider_native_typed` or `api`. Typed arguments
must equal the canonical body's three fields exactly; only the operation is
selected by the native function name.

SR supplies private history returned by the worker. Provider call IDs, reasoning
and other required native items are preserved privately. A nonterminal call needs
a canonical host result matching its operation, request ID and returned internal
call ID. Canonical-only assistant histories cannot reconstruct provider items and
are rejected. `agent_final` receives a local terminal acknowledgement without a
second model request. Changing projection requires a new private session.

## Verification

Run `pytest`, then exercise the
actual SR/provider route. Native Function Calling tests must not instruct the
model to write a textual API adapter wrapper instead of making a real tool call.
Keep the original task/context and task assertions, and verify the raw native
mapping independently. Report natural-language tasks separately from signed
host tasks; use only caller-owned requirements for host bindings. Preserve failed
attempts and verification reports locally, outside published repositories.
