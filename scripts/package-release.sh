#!/usr/bin/env bash
set -Eeuo pipefail

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd "$SERVICE_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/release/common.sh"

VERSION=""
TARGET=""
VARIANT="cpu"
LLAMA_SERVER=""
PYINSTALLER_VERSION=""
PIP_VERSION=""
PYTHON_VERSION=""
STAGING=""
OUTPUT=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --llama-server) LLAMA_SERVER="$2"; shift 2 ;;
    --pyinstaller-version) PYINSTALLER_VERSION="$2"; shift 2 ;;
    --pip-version) PIP_VERSION="$2"; shift 2 ;;
    --python-version) PYTHON_VERSION="$2"; shift 2 ;;
    --staging) STAGING="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    *) die "Unknown Models packaging argument: $1" 2 ;;
  esac
done

[ -n "$VERSION" ] && [ -n "$TARGET" ] && [ -n "$LLAMA_SERVER" ] &&
  [ -n "$PYINSTALLER_VERSION" ] && [ -n "$PIP_VERSION" ] && [ -n "$PYTHON_VERSION" ] && [ -n "$STAGING" ] && [ -n "$OUTPUT" ] ||
  die "Models packaging requires version, target, llama-server, runtime versions, staging and output" 2
validate_semver "$VERSION"
assert_local_target "$TARGET"
assert_inside_workspace "$STAGING"
assert_inside_workspace "$OUTPUT"
require_command python3
require_command node
require_file "$LLAMA_SERVER"
[ -x "$LLAMA_SERVER" ] || die "llama-server is not executable: $LLAMA_SERVER" 3
ACTIVE_PYTHON="$(python3 -c 'import platform; print(platform.python_version())')"
[ "$ACTIVE_PYTHON" = "$PYTHON_VERSION" ] || die "Models must be packaged with Python $PYTHON_VERSION (current: $ACTIVE_PYTHON)" 3
case "$VARIANT" in cpu|cuda|metal) ;; *) die "Unsupported Models variant: $VARIANT" 2 ;; esac
case "$TARGET:$VARIANT" in
  linux-*:cpu|linux-*:cuda|win32-*:cpu|win32-*:cuda|darwin-*:cpu|darwin-*:metal) ;;
  *) die "Models variant $VARIANT is not valid for target $TARGET" 2 ;;
esac

rm -rf "$STAGING"
mkdir -p "$STAGING" "$OUTPUT/components" "$OUTPUT/.metadata/$TARGET" "$OUTPUT/logs"
LOG_FILE="$OUTPUT/logs/models-${VARIANT}.log"
VENV="$STAGING/venv"

log_info "Creating isolated Models environment ($VARIANT)"
run_logged "$LOG_FILE" python3 -m venv "$VENV"
if [ -x "$VENV/bin/python" ]; then
  PYTHON="$VENV/bin/python"
else
  PYTHON="$VENV/Scripts/python.exe"
fi
run_logged "$LOG_FILE.pip" "$PYTHON" -m pip install "pip==$PIP_VERSION"
case "$VARIANT" in
  cpu) VARIANT_REQUIREMENTS="$SERVICE_DIR/requirements-cpu.txt" ;;
  cuda) VARIANT_REQUIREMENTS="$SERVICE_DIR/requirements-gpu.txt" ;;
  metal) VARIANT_REQUIREMENTS="$SERVICE_DIR/requirements-metal.txt" ;;
esac
run_logged "$LOG_FILE.requirements-variant" "$PYTHON" -m pip install -r "$VARIANT_REQUIREMENTS"
run_logged "$LOG_FILE.requirements" "$PYTHON" -m pip install -r "$SERVICE_DIR/requirements.txt"
run_logged "$LOG_FILE.dependencies-check" "$PYTHON" -m pip check
run_logged "$LOG_FILE.pyinstaller" "$PYTHON" -m pip install "pyinstaller==$PYINSTALLER_VERSION"
(cd "$SERVICE_DIR" && run_logged "$LOG_FILE.tests" "$PYTHON" -m unittest \
  tests/test_runtime_variant.py tests/test_setup_models.py)

log_info "Inspecting Models runtime"
RUNTIME_JSON="$STAGING/runtime.json"
"$PYTHON" "$SERVICE_DIR/scripts/inspect-runtime.py" --variant "$VARIANT" --llama-server "$LLAMA_SERVER" >"$RUNTIME_JSON"

log_info "Building Models PyInstaller bundle"
(cd "$SERVICE_DIR" && run_logged "$LOG_FILE.bundle" "$PYTHON" -m PyInstaller --noconfirm \
  --distpath "$STAGING/dist" --workpath "$STAGING/build" "$SERVICE_DIR/documents-models.spec")
BUNDLE="$STAGING/dist/documents-models"
[ -x "$BUNDLE/documents-models" ] || die "PyInstaller bundle entrypoint not found" 4
mkdir -p "$BUNDLE/llama"
cp -a "$(dirname "$LLAMA_SERVER")/." "$BUNDLE/llama/"

REVISION="$(git_revision "$SERVICE_DIR")"
node - "$BUNDLE/component-manifest.json" "$VERSION" "$TARGET" "$REVISION" "$VARIANT" "$RUNTIME_JSON" <<'NODE'
import fs from 'node:fs';
const [, , file, version, target, revision, variant, runtimeFile] = process.argv;
const inspectedRuntime = JSON.parse(fs.readFileSync(runtimeFile, 'utf8'));
fs.writeFileSync(file, `${JSON.stringify({
  schemaVersion: 1,
  component: 'models',
  version,
  target,
  variant,
  entrypoint: process.platform === 'win32' ? 'documents-models.exe' : 'documents-models',
  source: { repository: 'models', revision },
  runtime: inspectedRuntime,
  requires: { backendProtocol: 'execution/v1' },
  createdAt: new Date().toISOString(),
}, null, 2)}\n`);
NODE

log_info "Running frozen Models self-check"
DOCUMENTS_MODELS_VARIANT="$VARIANT" MODELS_DATA_DIR="$STAGING/self-check-data" \
  run_logged "$LOG_FILE.smoke" "$BUNDLE/documents-models" --self-check

EXT="tar.gz"
if [[ "$TARGET" == win32-* ]]; then EXT="zip"; fi
NAME="documents-models-v${VERSION}-${TARGET}-${VARIANT}.${EXT}"
ARTIFACT="$OUTPUT/components/$NAME"
if [ "$EXT" = "zip" ]; then
  require_command zip
  (cd "$BUNDLE" && zip -qr "$ARTIFACT" .)
else
  tar czf "$ARTIFACT" -C "$BUNDLE" .
fi

node - "$OUTPUT/.metadata/$TARGET/models-${VARIANT}.json" "$VERSION" "$TARGET" "$VARIANT" "components/$NAME" "$RUNTIME_JSON" <<'NODE'
import fs from 'node:fs';
const [, , file, version, target, variant, artifact, runtimeFile] = process.argv;
fs.writeFileSync(file, `${JSON.stringify({
  component: 'models', version, target, variant, artifact,
  runtime: JSON.parse(fs.readFileSync(runtimeFile, 'utf8')),
  requires: { backendProtocol: 'execution/v1' },
}, null, 2)}\n`);
NODE

log_info "Models artifact created: $ARTIFACT"
