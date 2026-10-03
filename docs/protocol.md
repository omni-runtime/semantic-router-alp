# Cloud ALP Function Calling

The optional `/v1/alp/chat/completions` endpoint projects ALP requests into native
Function Calling on explicitly enabled OpenAI-compatible cloud models. SR and
Envoy retain provider routing, credentials and HTTP dispatch. An operator-owned
local stdio worker supplies the authoritative ALP contracts and validates the
result. The worker does not call providers or execute agents or tools.

The worker requires access to the private `omni-runtime/alp_schema_mcp` repository.
Obtain dependency access separately. Its source, contract bundles and credentials
are not included in this public repository or its images. Install an authorized worker environment and mount it
read-only in the router container. The cloud worker imports the common protocol
dependencies but does not load vLLM, a GPU, a tokenizer or XGrammar.


Signed or catalog-bound tasks retain the smaller native schema view: fixed host
values and original union assertions remain intact, without duplicating union
properties. Unbound tasks expose the redundant type/property information.
`alp.native_schema_view` reports `bound_minimal`, `expanded` or `api_envelope`.
This affects schema presentation only; the canonical and host validators are
identical and no provider output is filled or repaired.

## Configuration

Use the canonical configuration's existing integration layer:

```yaml
global:
  integrations:
    alp:
      enabled: true
      command:
        - /opt/alp/bin/python
        - -m
        - semantic_router_alp.cloud_worker
        - --config
        - /etc/alp/catalogs.json
        - --projection
        - typed
      worker_env: [ALP_HOST_TASK_KEY]
      timeout_milliseconds: 10000
      models:
        cloud/chat:
          enable_thinking: false
          allow_stop_tool_call: true
```

The command and environment allowlist are operator configuration; clients cannot
override them. Provider API credentials are never passed to the worker. Configure
provider URLs and credentials in `providers.models`; declare the `tools` model
capability in `routing.modelCards`, alongside the model's other capabilities.
The ALP integration's `models` contains allowed concrete backend names. Requests
can use existing public entrypoints (for example `cloud-only`); normal SR routing
selects a backend, and the ALP allowlist is checked again before dispatch. The
global entrypoint policy remains in force. A private replay cannot switch the
selected provider model, and router agent-loop execution is rejected for ALP.
Ordinary inference endpoints are unchanged when this integration is disabled.

Set provider quirks only after verifying the selected backend. Qwen-compatible
thinking mode may reject forced `tool_choice`; `enable_thinking: false` disables
it for this ALP route. `allow_stop_tool_call` permits providers which return
`stop` for a complete function call. It still requires exactly one valid call;
empty output, text mixed with calls, refusal, truncation and parallel calls fail.

## Request and validation

Requests use the shared `alp_schema_mcp.runtime` ALP chat DTO (`model`, `messages`, `alp`, generation
controls). The six native function names come from `alp_schema_mcp`. The worker
supports two operator-selected projections:

- `api_json` (default): the MVP API envelope has `protocol_version`, `request_id`
  and a once-encoded `payload_json` string. The worker requests `strict: true` for
  this envelope; even enforced outer fields do not validate the string's content.
- `typed`: arguments have `protocol_version`, `request_id` and object `payload`.
  The schema is compiled from the same operation, catalog and host constraints.
  The ALP protocol permits operation-specific static native projections. The
  adapter adds the operation from the authoritative function-name mapping and
  validates the canonical request without filling or rewriting payload fields.

Typed schemas expose implied types and closed-union properties for native tool
consumers while retaining the original reference and union assertions. Host
schema literals stay exact. The native parameters are the single schema source;
no duplicate action schema is added to the system context in typed mode.

Typed mode does not request provider strict mode by default. Operators may add
`--request-strict` after evaluating provider compatibility. Accepting the flag
is not evidence that the provider enforces the full ALP schema. Responses report
`provider_strict_enforcement: not_requested` or `not_verified`, never a guarantee.
Use a new private session when switching projections.

After generation, the worker decodes the native call once and applies the full
canonical protocol, model-visible catalog, dynamic arguments, host contracts and
signed task constraints. Duplicate JSON keys, unknown fields and malformed native
arguments are rejected. Whitespace-only assistant content is treated as framing
and retained in private history; mixed explanatory text is rejected.
No tagged-content scanning, automatic tool execution,
output repair or hidden model retry occurs. Successful responses have
`alp.validated: true`, `alp.executed: false` and `alp.authorized: false`.

Use `alp_schema_mcp.runtime.host_tasks.AgentCallTask` or `DefinitionTask` and
`host_task_headers(request, task, key=...)` for exact caller-owned task text,
session modes, artifacts and capability interfaces. SR accepts the existing
`x-alp-task-context` header, validates it before dispatch, and removes it before
the cloud request. The signing key must match the worker catalog configuration's
`task_signing_key_env`. Coverage metadata distinguishes fixed requirements from
requirements still left to model interpretation. An unbound natural-language
request can pass protocol validation while failing an intended task requirement.
For typed tasks, requirements are native parameter assertions, including fixed
values and capability schemas. In `api_json` mode a bound payload schema is
attached to the `payload_json` description; unbound requests retain the full
action context. Neither approach guarantees provider compliance.

Definitions may bind `environment_profile_ref`, `requested_tools`, resource
requirements and direct `output` even without named capabilities. Choose direct
`output` or `output_from_capability`, not both. These values must come from the
caller's task contract, not an evaluation oracle.

With `include_raw: true`, `raw` contains the original native function name and
argument string. `alp.raw_codec` is `provider_native_typed` or `api`; consult it
before decoding. Typed raw arguments must equal the canonical body's three fields
exactly. `alp.provider_projection` records the configured projection.

## Private replay

Configure caller authentication on the gateway. SR binds private replay state to
the incoming bearer credential, logical model and catalog. Each accepted response
returns `alp.provider_state`, an opaque random handle. Send it in
`x-alp-provider-state` on the next turn with only the new messages. Keep this handle
in the host's private session; do not forward it to other agents.

For a nonterminal call, the next turn starts with a `tool` message whose
`tool_call_id` is the returned internal `agent_calls[0].id`, and whose content is
the canonical ALP result. The worker validates its operation and request ID, then
maps it to the original provider call ID. Unpaired and mismatched results fail.
Provider-required reasoning and native items remain in the private store and are
replayed without exposing them in the public ALP response. Canonical-only
assistant call histories are rejected because they cannot reconstruct these items.

For `agent_final`, the adapter adds a paired terminal acknowledgement locally.
It makes no additional model request. Failed or truncated candidate calls are not
added to replay history. The host still owns authorization, action execution,
durable runs, effect accounting and execution-result persistence.

Replay storage is process-local: 30-minute expiry, 256 snapshots, a 32 MiB total
bound and an 8 MiB per-snapshot bound. Restart, expiry, a different caller or an
unknown handle returns an explicit error. Use a single router replica or sticky
routing for this mode; this implementation does not claim durable or shared replay.
Response caching, general router replay capture and automatic model fallback are
disabled for ALP calls, preserving private provider state and single-generation
semantics.

## Streaming and testing

The provider response is buffered until validation. `stream: true` returns SSE
with a validated `agent_call.completed` event, rather than provisional token
deltas. Invalid buffered responses produce an error and never a completed event.
The worker has bounded input/output, an explicit timeout and at most eight
concurrent processes.

Run ALP core tests in the authorized dependency checkout, Go transport and config
tests, and the `TestNativeALP` ExtProc tests with the repository's native libraries.
Then validate the deployed Envoy/SR path against the actual provider, including
HTTP error status, truncation, signed task binding, session pairing and caller
isolation. Keep deployment credentials and verification reports outside Git.

The native provider verifier `scripts/verify-alp.py` preserves the producer task,
context and task assertions. By default it omits the suite's API adapter-object
format instruction: that instruction tests a textual `{name, arguments}` wrapper,
whereas this path tests real provider Function Calling. It checks canonical
protocol/task assertions and independently checks the exact native-to-canonical
mapping. Use `--adapter-object-instructions` only for explicit legacy comparisons.
Keep original natural-language scores separate from signed host-task scores;
the latter provide requirements through the trusted caller interface. Neither
mode silently retries failed generations or supplies golden outputs to the model.
