#!/usr/bin/env bash
# Build Linux natives for the host arch inside docker (native/Dockerfile), export them
# to native/dist, and sync the SWIG-generated Java into src/main/java.
# Linux is the canonical leg for the generated Java (pinned SWIG version in the Dockerfile).
#
#   native/linux.sh                  build + sync generated java into src/main/java
#   native/linux.sh --no-sync-java   build only (what CI runs)
#   env DOCKER_BUILD_EXTRA_ARGS      extra `docker build` args (CI uses it for cache flags)
#
# Output:
#   native/dist/resources/lightgbm4j/linux/<arch>/lib_lightgbm.so{,.md5}
#   native/dist/resources/lightgbm4j/linux/<arch>/lib_lightgbm_swig.so{,.md5}
#   native/dist/java/com/microsoft/ml/lightgbm/*.java
set -euo pipefail

NATIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$NATIVE")"
DIST="$NATIVE/dist"
JAVA_DST="$ROOT/src/main/java/com/microsoft/ml/lightgbm"

SYNC=1
[[ "${1:-}" == "--no-sync-java" ]] && SYNC=0

[[ -d "$NATIVE/lightgbm/external_libs/eigen/Eigen" ]] \
  || { echo "LightGBM submodule missing: run 'git submodule update --init --recursive'" >&2; exit 1; }

# shellcheck disable=SC2086
docker build -f "$NATIVE/Dockerfile" --target dist --output "type=local,dest=$DIST" \
  ${DOCKER_BUILD_EXTRA_ARGS:-} "$ROOT"

find "$DIST/resources" -type f | sort

if [[ $SYNC == 1 ]]; then
  # PredictionType.java and SwigPointers.java are hand-written and live next to the generated files
  find "$JAVA_DST" -maxdepth 1 -name '*.java' ! -name 'PredictionType.java' ! -name 'SwigPointers.java' -delete
  cp "$DIST"/java/com/microsoft/ml/lightgbm/*.java "$JAVA_DST/"
  echo "synced generated SWIG java into $JAVA_DST"
fi
