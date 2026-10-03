# Build the composed router

Obtain access to the private contract dependency and install its locked revision.
The public source and patch checks do not need that access.

```bash
python scripts/prepare.py --output .work/prepared
bash scripts/test-native.sh .work/prepared
```

An existing upstream clone can be passed with `--source`; an existing multimodal
repository can be passed with `--multimodal`. Preparation reads the **locked Git
commit**, not that repository's working files. It checks both patch manifests and
the final composed source fingerprint. Existing output directories are rejected.

Build the resulting upstream checkout using the locked multimodal project's native
build toolchain and image instructions. Its base commit is recorded in
`dependencies.lock.json`. Full SR tests/builds need CGO and the upstream native
libraries. Cross-target test execution must explicitly supply `GO_TEST_EXEC`.
The cloud Python worker itself requires no native inference libraries.

`release.yaml` retains the historical preview artifact and its original source
composition; it does not identify a rebuilt image for the current source tree.
Build current sources from `dependencies.lock.json` and `patches/series`, and
record a new operator-owned release contract before deployment. Historical local OCI references
are provenance identifiers, not publicly available registry downloads. When the
composed Go source is identical, an existing verified router binary can be reused;
the separately mounted Python worker still needs the new package and contract
runtime. Install private dependencies only in an authorized worker environment,
mount it read-only, and configure the entrypoint in `inference-stack`.

The operator remains responsible for provider credentials and caller authentication.
Worker processes receive only the configured environment allowlist. No credential,
private contract bundle, deployment report or fleet inventory belongs in an image
published from this repository.
