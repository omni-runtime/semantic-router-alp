# Changelog

## 0.2.0

- Add opt-in ALP 0.4 response collections with atomic validation and paired terminal results.
- Keep 0.3 as the default; isolate catalogs and private replay by protocol version.
- Enforce reserved resource management, exclusive actions and per-response IDs.
- Update the producer runner for the latest 0.3/0.4 suites and pin the authorized contract revision.
- Keep native cloud schemas independent of local decoder ordering, preserving optional dependencies and explicit host constraints.
- Clarify definition interfaces and collection semantics; present scalar constants as equivalent singleton enums without rewriting generated actions.
- Add explicit, bounded format repair with private provider-call pairing; report validation scope separately from task coverage.

## 0.1.1

Keep catalog specialization, task binding and validation helpers inside this plugin.
Use the unchanged alp-schema-mcp 0.3.0 dependency; no alp-core package or inference
plugin is required. Native cloud projection and SR transport behavior are preserved.

## 0.1.0

Extract cloud ALP adaptation and SR transport integration into a dedicated repository.
The initial shared-runtime dependency is superseded by 0.1.1.
