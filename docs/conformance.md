# ALP protocol correspondence

Protocol authority: private `omni-runtime/alp_schema_mcp`, ALP 0.3.0 draft 1.
Its `contracts/README.md` section 12 and `transport-map.json` define provider-native
adaptation. Contract files are loaded from that dependency and are never copied here.
This document maps requirements to code and tests; it is not a deployment report.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Six authoritative function mappings | `CloudAdapter.prepare/complete`, ContractCatalog | Six-operation tests in `tests/test_cloud.py` |
| Decode payload_json exactly once; validate canonical and dynamic arguments | `CloudAdapter.complete`, shared ALPParser | Bad payload, double encoding, wrong target and host binding tests |
| One parsing mode and one complete action | Native completion validation | Mixed content, parallel, empty, refusal and truncation tests |
| Static typed projection preserves canonical meaning | `cloud_projection.py` | Raw equality, schema invariance and bound literal tests |
| Separate provider call ID, internal call ID and request_id | Adapter context and private history | Exact paired-result tests |
| Preserve required provider history privately | Adapter history plus SR ReplayStore | Reasoning retention; caller/catalog isolation and expiry tests |
| Terminal acknowledgement without another generation | agent_final completion | Terminal pair and replay tests |
| No execution or authorization from validation | Response flags; no executor | All operation responses have authorized/executed false |
| Trusted task constraints cannot be supplied by model text | Shared signed host-task binding | Signature and task-drift tests |
| Buffer until complete validation before SSE success | SR `processor_alp.go` | `TestNativeALP` and deployed streaming/truncation tests |
| Bound local IPC and environment; no provider secret in worker | SR `alptransport/worker.go` | Worker limits/timeout/environment tests |

The authoritative parser remains the final check even when provider parameters use
strict mode. Typed projection can express more of a bound task but does not promise
that a provider enforces every JSON Schema assertion. A protocol-valid response can
still miss an unbound user requirement; deployed scoring reports both separately.

This integration implements generation and protocol adaptation. The host must still
perform real identity/permission checks, execution, effect accounting and durable
Run management before acting on an accepted call. Process-local replay is not a
durable execution ledger and is not advertised as one.
