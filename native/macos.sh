#!/usr/bin/env bash
# Build macOS natives for the host arch (x86_64 or arm64).
# Needs: `brew install cmake swig libomp` and a JDK (javac on PATH).
#
# Output:
#   native/dist/resources/lightgbm4j/osx/<arch>/lib_lightgbm.dylib{,.md5}
#   native/dist/resources/lightgbm4j/osx/<arch>/lib_lightgbm_swig.dylib{,.md5}
#   native/dist/java/com/microsoft/ml/lightgbm/*.java
set -euo pipefail

NATIVE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$NATIVE/lightgbm"
BUILD="$SRC/build"
DIST="$NATIVE/dist"

# deployment targets match what upstream LightGBM uses for its own macOS artifacts
case "$(uname -m)" in
  x86_64) ARCH=x86_64;  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-10.15}" ;;
  arm64)  ARCH=aarch64; export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-12.0}" ;;
  *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;;
esac

[[ -d "$SRC/external_libs/eigen/Eigen" ]] \
  || { echo "LightGBM submodule missing: run 'git submodule update --init --recursive'" >&2; exit 1; }
for t in cmake swig javac; do
  command -v "$t" >/dev/null || { echo "$t not found on PATH (brew install cmake swig; install a JDK)" >&2; exit 1; }
done
brew --prefix libomp >/dev/null 2>&1 || { echo "libomp missing: brew install libomp" >&2; exit 1; }
export JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home)}"
echo "building osx/$ARCH with JAVA_HOME=$JAVA_HOME, $(swig -version | grep -i version)"

cmake -B "$BUILD" -S "$SRC" -DUSE_SWIG=ON -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --config Release -j "$(sysctl -n hw.ncpu)"

# upstream's POST_BUILD step stages both libs here with the .jnilib already renamed to .dylib
# (path is hard-coded x86_64 upstream, even on arm64)
STAGE="$BUILD/com/microsoft/ml/lightgbm/osx/x86_64"
OUT="$DIST/resources/lightgbm4j/osx/$ARCH"
mkdir -p "$OUT" "$DIST/java/com/microsoft/ml/lightgbm"
for lib in lib_lightgbm lib_lightgbm_swig; do
  cp -f "$STAGE/$lib.dylib" "$OUT/$lib.dylib"
  printf '%s' "$(md5 -q "$OUT/$lib.dylib")" > "$OUT/$lib.dylib.md5"
done
rm -f "$DIST"/java/com/microsoft/ml/lightgbm/*.java
cp "$BUILD"/java/*.java "$DIST/java/com/microsoft/ml/lightgbm/"

ls -l "$OUT"
otool -L "$OUT/lib_lightgbm.dylib"
