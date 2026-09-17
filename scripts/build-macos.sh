#!/usr/bin/env bash
#
# Build and test on macOS, in one command.
#
#   scripts/build-macos.sh                 configure, build and test (debug)
#   scripts/build-macos.sh release         the release preset
#   scripts/build-macos.sh asan            debug plus sanitizers
#   scripts/build-macos.sh debug --clean   throw the build directory away first
#   scripts/build-macos.sh debug --no-test build only
#
# The point of this script is the failure path: a compile error or a failing
# test is printed on its own, at the end, without the hundred lines of cmake
# chatter around it, so it can be pasted somewhere useful. Everything it hides
# is still in build/<preset>/build.log.

set -uo pipefail

preset="${1:-debug}"
shift || true

case "${preset}" in
    debug | macos-debug) preset="macos-debug" ;;
    release | macos-release) preset="macos-release" ;;
    asan | macos-asan) preset="macos-asan" ;;
    core | core-debug) preset="core-debug" ;;
    *)
        echo "unknown preset '${preset}' (try: debug, release, asan, core)" >&2
        exit 2
        ;;
esac

clean=0
run_tests=1
for argument in "$@"; do
    case "${argument}" in
        --clean) clean=1 ;;
        --no-test) run_tests=0 ;;
        *)
            echo "unknown option '${argument}'" >&2
            exit 2
            ;;
    esac
done

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

build_dir="build/${preset}"
log="${build_dir}/build.log"

if [[ "${clean}" == 1 ]]; then
    echo "==> removing ${build_dir}"
    rm -rf "${build_dir}"
fi

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" != "arm64" ]]; then
    echo "This project is Apple silicon only (ADR-002): mesh shaders need an" >&2
    echo "Apple7 GPU, which no Intel Mac has." >&2
    exit 1
fi

for tool in cmake ninja; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "${tool} is not installed. brew install cmake ninja" >&2
        exit 1
    fi
done

mkdir -p "${build_dir}"

echo "==> configuring (${preset})"
if ! cmake --preset "${preset}" >"${log}" 2>&1; then
    echo "--- configure failed ---"
    tail -40 "${log}"
    echo "--- full log: ${log} ---"
    exit 1
fi

echo "==> building"
if ! cmake --build --preset "${preset}" >>"${log}" 2>&1; then
    echo "--- build failed ---"
    # Compiler diagnostics, and the two lines of context around each, which is
    # usually the whole story.
    grep -E -A2 "(error|Error):" "${log}" | head -60
    echo "--- full log: ${log} ---"
    exit 1
fi

if [[ "${run_tests}" == 0 ]]; then
    echo "==> built (tests skipped)"
    exit 0
fi

echo "==> testing"
test_preset="${preset}"
case "${preset}" in
    macos-release) test_preset="macos-debug" ;; # no test preset for release
    macos-asan) test_preset="macos-debug" ;;
esac

if ! ctest --test-dir "${build_dir}" --output-on-failure -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    >"${build_dir}/test.log" 2>&1; then
    echo "--- tests failed ---"
    # Catch2 prints the assertion, the expansion and the file:line; that block
    # is what someone else needs to see.
    sed -n '/FAILED/,/^$/p' "${build_dir}/test.log" | head -60
    echo
    grep -E "tests passed|tests failed" "${build_dir}/test.log" | tail -2
    echo "--- full log: ${build_dir}/test.log ---"
    exit 1
fi

tail -3 "${build_dir}/test.log"
echo "==> ${preset}: green"
