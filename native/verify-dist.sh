#!/usr/bin/env bash
# Fails unless native/dist/resources holds all 10 native binaries and 10 md5 sidecars.
# Used by the CI fan-in job before packaging the jar.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dist/resources/lightgbm4j"
missing=0
for leg in linux/x86_64:so linux/aarch64:so osx/x86_64:dylib osx/aarch64:dylib windows/x86_64:dll; do
  dir="${leg%%:*}"
  ext="${leg##*:}"
  for lib in lib_lightgbm lib_lightgbm_swig; do
    f="$ROOT/$dir/$lib.$ext"
    [[ -s "$f" ]] || { echo "MISSING $f"; missing=1; continue; }
    [[ "$(wc -c < "$f.md5" | tr -d ' ')" == 32 ]] || { echo "BAD MD5 SIDECAR $f.md5"; missing=1; }
  done
done
if [[ $missing == 0 ]]; then
  echo "all 10 natives present"
else
  exit 1
fi
