# Opt-in ALP 0.4

ALP 0.3 remains the default. Set `alp.protocol_version` to `"0.4.0"` explicitly
for response collections. The private `alp_schema_mcp` dependency must be installed
from commit `459257275b29865d3f60facb67283c8bfab1262e` (see the dependency lock).
Its package version is still 0.3.0; the version number alone does not select this
contract revision. Obtain dependency access separately. No standalone core
package or service is required.

The six operations are unchanged. One completed model turn contains 1–16 ordinary
requests, with unique request IDs. Native Function Calling preserves each provider
call ID privately; canonical responses are arrays even for one item, and tagged
responses concatenate ordinary frames. `agent_final` and the reserved
`resource.bindings.update` tool each require a singleton response. The reserved
management descriptor cannot be replaced or delegated by a generated definition.

The public completion retains `choices[0].message.agent_calls`, now with one entry
per member. All entries are published together after whole-response validation.
SSE deltas remain provisional text, never executable actions. Failure, cancellation,
truncation, duplicate IDs, trailing content or any invalid member rejects the whole
response. The complete ALP projection is limited to 128 KiB.

`alp.validation_scope` and `alp.task_constraint_coverage` are ordered per-member
arrays in 0.4; they retain their object shape in 0.3. Catalog task constraints still
apply to every member of the selected operation. Use response `members` for
different per-member requirements in one turn; they intersect the global
constraints. Protocol validation does not claim natural-language task matching.

Return one canonical terminal Result per accepted member, in original request
order, using that member's public call ID as `tool_call_id`. All results must be
available before requesting the next model turn. Count, request ID, operation and
successful target are checked by `validate_action_exchange`. Provider-private
replay also binds the protocol version and projection. A final receives only a
local provider acknowledgment and closes replay; it never creates an ALP success
Result or permits another generation in that session.

The adapter does not execute tools, register agents, grant permissions, manage
Run/resource revisions, or schedule actions in parallel. These remain Runtime
responsibilities. Static protocol and producer tests do not validate those Runtime
features. No public batch operation or success envelope is introduced.

## Reproducible producer tests

Use the latest authorized `test-case` checkout. The runner supports both protocol
versions and no longer depends on the removed per-case `/operation` assertion.
Supply an operator-owned `--case-config` JSON map when cases require different
visible catalogs. Each entry contains `operations` and `catalog_ref`; expected
answers are used only by the suite scorer after generation.

```bash
export ALP_API_KEY=...  # use a private environment, never commit credentials
python scripts/verify-alp.py --suite /path/to/test-case \
  --base-url http://your-endpoint --model your-model \
  --protocol-version 0.4.0 --case-config /private/cases.json \
  --output /private/reports/run-04
```

Reports distinguish endpoint acceptance from exact task matching, retain raw wire
output, and report failures without retries by default. `--repair-once` explicitly
consumes an available format-repair token and records both attempts and retry count. Run again with
`--protocol-version 0.3.0` for the original 20-case regression. Testing does not
execute any generated call. Keep reports, endpoint keys and deployment catalogs
outside the public repository.

## Trusted response constraints

For ALP 0.4, a catalog may set `response_constraints` with `min_calls` and
`max_calls` (1–16), or an ordered `members` list. Each member names an operation
and supplies its own `PayloadConstraints`; its position is significant. A member
list fixes the response count. Global operation constraints remain in force.
`forbidden_fields` contains object JSON Pointers under payload and prevents
unsupplied host values such as `expected_state_version` from being invented.

Trusted callers can use `host_response_headers` to sign the same member plan
as task-context version 2. The signature covers the entire request and expires;
it cannot widen catalog limits, replace fixed values, or change the operation.
Existing version-1 operation constraints remain supported. ALP 0.3 rejects
response-level constraints. Neither mechanism grants permission or executes calls.

`DefinitionTask.output_from_capability` derives the final output schema from the
same capability contract. Use it with `capabilities` to avoid independently
generating two interface schemas. An omitted environment means the protocol
default environment; `default` is not an implicit catalog profile name.

Responses expose `response_constraint_coverage` plus per-call task coverage.
Protocol validity does not claim that unbound natural-language requirements were
met. Producer evaluations should report original prompt-only cases separately
from tests with trusted task inputs bound at the host. Tool execution is outside
the producer evaluation.

## Explicit cloud format repair

A completed native Function Calling response rejected for ALP format or bound
host constraints may return an error with `alp.provider_state` and
`alp.format_repair_available: true`. No failed action is accepted or executed.
To request one repair, repeat exactly the original request and task constraints
with this token in `x-alp-provider-state`. The token is private, scoped to the
authenticated caller/model/catalog, expires after 30 minutes, and is consumed
on retrieval (including a rejected changed request). Concurrent or repeated
use fails with HTTP 409. A second invalid answer has no further repair token.

SR privately retains the original provider call IDs and pairs each failed call
with a provider-local validation error. These are not canonical ALP Results.
Malformed JSON arguments are retained only as private diagnostics, never replayed
as native calls. Their single explicit repair starts from the original task plus
local format diagnostics. Valid JSON calls rejected by the contract retain paired
provider-local errors. Truncated responses and unpairable IDs cannot be repaired this way. No hidden
retry occurs. Typed provider schemas describe member alternatives; they do not
guarantee ordering or count. SR validates both after generation.

## Framework semantic presentation and observed state

The default `semantic_rendering: true` catalog option adds concise protocol
semantics and preserves the original user task. It contains no fixture answers,
exact response counts or inferred permissions. `false` disables these hints for
controlled comparisons. Typed function schemas inline only equivalent small leaf
references; tool argument schemas are not repeated in the tool-call system context.
Definition generation still receives tool interfaces needed for its capabilities.

Cloud projection uses protocol/host schemas independently of local decoder field
ordering. `requested_tools` and `resource_requirements` remain optional (default
`[]`) unless a trusted constraint explicitly requires them. Explicit operator
output and prose limits remain enforced. This applies to typed native parameters
and the schema description accompanying `payload_json`; no defaults are inserted
into the returned model action.

Field descriptions distinguish the whole Agent input from capability arguments,
preserve explicitly supplied names/list order, and explain that a tool reference
does not itself create a resource slot. Multiple definitions may be requested in
one completed model response; serial Runtime registration does not impose a
one-definition generation limit. Calling new instances still requires observing
their returned IDs in a later turn.

Typed native parameters present scalar constants as equivalent singleton enums
for provider compatibility. This does not decode or coerce returned values: a
version string containing embedded JSON quotes is still rejected. Fixed object
values and embedded business-schema literals are left untouched, and the original
canonical ALP validator remains authoritative.

`Capability.state_effect`/`external_effect`, `Agent.state_version` and
`Catalog.current_state_version` accept trusted Runtime metadata. A declared
state-writing tool/capability without its observed version fails preparation with
`MISSING_RUNTIME_CONTEXT`; with a version, generation and validation require that
exact precondition. Read calls retain optional explicit version guards. Supply
fresh request-scoped metadata through catalog resolution; a configured snapshot
does not replace Runtime's atomic execution-time version check.

`ServerConfig.native_tool_choice_policy` supports `auto`, `required`, and `named`.
The default `auto` preserves existing selection behavior; `named` requires a single
allowed function. Provider support must be measured, not inferred from OpenAI API
compatibility. `strict: true` remains a request, not proof of provider enforcement.
Successful responses expose sanitized `alp.provider_finish_reason`,
`alp.provider_tool_choice`, and `alp.semantic_rendering` for reproducible diagnosis.

### Signing an ordered host request

The catalog must already expose every referenced target. With an ALP 0.4
`ALPChatRequest` selecting `agent_call`, the trusted caller can sign a plan:

```python
from semantic_router_alp.host_tasks import host_response_headers

headers = host_response_headers(
    request,
    members=[
        {"operation": "agent_call", "payload": {
            "fixed_values": {"/instance_id": target, "/input/task": task,
                             "/session_mode": "isolated"},
            "forbidden_fields": ["/expected_state_version"],
        }}
        for target, task in approved_tasks
    ],
    key=trusted_host_signing_key,
)
# Send request.model_dump() and these headers to /v1/alp/chat/completions.
```

`approved_tasks` is supplied by the host. Do not infer permission, target IDs, or
a signing key from model output. Counts and task values are enforced; descriptions
and other requirements left unbound remain the model's responsibility.

Ordered native tool schemas expose the body fields directly at the root. Shared
definitions and common assertions are factored without weakening validation,
so repeated calls do not unnecessarily exhaust the router's context budget.

The single format-repair turn receives structured local diagnostics, including
required/actual call counts and missing envelope fields. These are provider-local
errors with `executed: false`, not canonical ALP execution results.
