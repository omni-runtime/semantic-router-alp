# ALP patch ownership

`0001-native-alp.patch` applies after the exact multimodal base recorded in
`dependencies.lock.json`. It owns only SR ALP transport, private replay,
configuration, validation and tests. Do not copy the base patch into this repo.

To export changes from a prepared checkout, diff against the tree after applying
the locked multimodal patch. Update this patch, its SHA-256 and the final composed
source fingerprint together. Run `scripts/prepare.py` into a fresh directory and
native tests before changing a deployed image. Upstream licenses continue to apply.
