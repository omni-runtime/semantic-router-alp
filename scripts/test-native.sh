#!/usr/bin/env bash
set -euo pipefail
checkout=${1:?Pass the prepared upstream checkout}
cd "$checkout/src/semantic-router"
# Use the same native CGO libraries and toolchain as the locked SR build.
# GO_TEST_EXEC may identify an explicitly configured cross-target runner.
runner=()
if [[ -n ${GO_TEST_EXEC:-} ]]; then runner=(-exec "$GO_TEST_EXEC"); fi
go test "${runner[@]}" ./pkg/alptransport -count=1
go test "${runner[@]}" ./pkg/config ./pkg/configschema -run 'Test.*ALP' -count=1
go test "${runner[@]}" ./pkg/extproc -run 'TestNativeALP' -count=1
