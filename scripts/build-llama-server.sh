#!/usr/bin/env bash
set -Eeuo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd "$SERVICE_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/release/common.sh"

VARIANT=""
REPOSITORY=""
REVISION=""
STAGING=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --repository) REPOSITORY="$2"; shift 2 ;;
    --revision) REVISION="$2"; shift 2 ;;
    --staging) STAGING="$2"; shift 2 ;;
    *) die "Unknown llama-server build argument: $1" 2 ;;
  esac
done

[ -n "$VARIANT" ] && [ -n "$REPOSITORY" ] && [ -n "$REVISION" ] && [ -n "$STAGING" ] ||
  die "llama-server build requires variant, repository, revision and staging" 2
case "$VARIANT" in cpu|cuda|metal) ;; *) die "Unsupported llama-server variant: $VARIANT" 2 ;; esac
assert_inside_workspace "$STAGING"
require_command cmake
require_command git
require_command make
require_command c++

case "$VARIANT" in
  cuda) require_command nvcc ;;
  metal) [ "$(uname -s)" = "Darwin" ] || die "Metal llama-server builds require macOS" 3 ;;
esac

SOURCE_DIR="$STAGING/source"
BUILD_DIR="$STAGING/build"
rm -rf "$STAGING"
mkdir -p "$STAGING"

log_info "Fetching pinned llama.cpp revision $REVISION"
git clone --filter=blob:none "$REPOSITORY" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --detach "$REVISION"
[ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$REVISION" ] || die "llama.cpp revision verification failed" 5

CMAKE_ARGS=(
  -S "$SOURCE_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DLLAMA_CURL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_UI=OFF
  -DLLAMA_USE_PREBUILT_UI=OFF
)
case "$VARIANT" in
  cuda) CMAKE_ARGS+=(-DGGML_CUDA=ON -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler) ;;
  metal) CMAKE_ARGS+=(-DGGML_METAL=ON) ;;
esac

log_info "Building llama-server ($VARIANT)"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target llama-server --parallel "$(getconf _NPROCESSORS_ONLN)"
SERVER="$BUILD_DIR/bin/llama-server"
[ -x "$SERVER" ] || die "llama-server build did not produce an executable" 5
