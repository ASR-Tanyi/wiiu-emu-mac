#!/usr/bin/env bash
#
# What this Mac can build, in one command.
#
#   scripts/check-macos.sh
#
# Phases 1.4, 1.8 and 7 need things the platform-independent core does not:
# the Metal compiler, a signing identity, an Apple7 GPU. Finding out which of
# them are missing one build failure at a time is slow, so this asks all the
# questions at once and writes the answers to build/toolchain-report.txt --
# inside the repository, where whoever is helping can read it without anyone
# pasting anything.
#
# It reports identities by *kind and count*, never by name: what matters is
# whether a Developer ID certificate exists, not whose it is.

set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

mkdir -p build
report="build/toolchain-report.txt"

{
    echo "wiiu-emu toolchain report -- $(date -u '+%Y-%m-%d %H:%M UTC')"
    echo

    echo "== host =="
    if [[ "$(uname -s)" != "Darwin" ]]; then
        echo "  not macOS ($(uname -s)) -- only the core-debug preset applies here"
    else
        echo "  macOS $(sw_vers -productVersion) ($(sw_vers -buildVersion)), $(uname -m)"
        if [[ "$(uname -m)" != "arm64" ]]; then
            echo "  ** Intel: ADR-002 rules this out. Mesh shaders need an Apple7 GPU."
        fi
        # The GPU decides whether the Phase 8.8 geometry path can exist at all.
        chip="$(system_profiler SPDisplaysDataType 2>/dev/null |
            awk -F': ' '/Chipset Model/ { print $2; exit }')"
        echo "  gpu: ${chip:-unknown}"
    fi
    echo

    echo "== xcode =="
    if ! command -v xcode-select >/dev/null 2>&1; then
        echo "  ** no xcode-select: no developer tools at all"
    else
        selected="$(xcode-select -p 2>/dev/null || echo '<none>')"
        echo "  selected: ${selected}"
        if [[ "${selected}" == *"CommandLineTools"* ]]; then
            echo "  ** Command Line Tools only."
            echo "     Enough to compile C++ and Objective-C++, so Phase 1.4 can start;"
            echo "     but there is no Metal compiler and no frame capture (7.8), so"
            echo "     install Xcode and: sudo xcode-select -s /Applications/Xcode.app"
        fi
        if command -v xcodebuild >/dev/null 2>&1; then
            echo "  $(xcodebuild -version 2>/dev/null | tr '\n' ' ')"
        fi
        echo "  macosx SDK: $(xcrun --show-sdk-version 2>/dev/null || echo '<none>')"
    fi
    echo

    echo "== metal toolchain =="
    # Phase 8.6 emits MSL text and compiles it; without this there is no
    # renderer beyond a cleared window.
    if metal_version="$(xcrun -sdk macosx metal --version 2>&1)"; then
        echo "  ${metal_version}" | head -2 | sed 's/^/  /'
    elif [[ "${selected:-}" == *"CommandLineTools"* || -z "${selected:-}" ]]; then
        echo "  ** no metal compiler: the Command Line Tools do not include one."
        echo "     Install Xcode, then: sudo xcode-select -s /Applications/Xcode.app"
    else
        # Xcode 26 unbundled the Metal toolchain: Xcode is installed and
        # selected, and `metal` still is not there until it is fetched. Apple's
        # own error says exactly this, so repeat their command rather than
        # inventing one.
        echo "  ** no metal compiler, but Xcode is selected -- the toolchain is"
        echo "     a separate component from Xcode 26 onwards:"
        echo "       xcodebuild -downloadComponent MetalToolchain"
        echo "     (or Xcode > Settings > Components). Details reported below:"
        echo "${metal_version}" | head -3 | sed 's/^/       /'
    fi
    echo

    echo "== build tools =="
    for tool in cmake ninja ccache git python3; do
        if command -v "${tool}" >/dev/null 2>&1; then
            printf '  %-8s %s\n' "${tool}" "$("${tool}" --version 2>&1 | head -1)"
        else
            printf '  %-8s ** missing\n' "${tool}"
        fi
    done
    echo "  (cmake and ninja are required: brew install cmake ninja)"
    echo

    echo "== signing (Phase 1.8) =="
    # Two separate questions, and they have different answers. Running a JIT
    # locally needs only an ad-hoc signature with the entitlement. Notarizing
    # needs a Developer ID certificate, which needs a paid membership.
    if command -v codesign >/dev/null 2>&1; then
        echo "  codesign: present (ad-hoc signing works -- that is all the JIT"
        echo "            entitlement needs to be proved locally)"
    else
        echo "  ** no codesign"
    fi
    if command -v security >/dev/null 2>&1; then
        identities="$(security find-identity -v -p codesigning 2>/dev/null || true)"
        developer_id="$(printf '%s' "${identities}" | grep -c 'Developer ID Application' || true)"
        total="$(printf '%s' "${identities}" | grep -cE '^ +[0-9]+\)' || true)"
        echo "  code-signing identities: ${total:-0} (Developer ID Application: ${developer_id:-0})"
        if [[ "${developer_id:-0}" == "0" ]]; then
            echo "     No Developer ID certificate, so notarization is not possible yet:"
            echo "     the notary service rejects ad-hoc and local-development signatures."
            echo "     That needs a paid Apple Developer Program membership. The ad-hoc"
            echo "     half of 1.8 can still be done now, and is the half the JIT needs."
        fi
    fi
    if command -v xcrun >/dev/null 2>&1; then
        if xcrun --find notarytool >/dev/null 2>&1; then
            echo "  notarytool: present"
        else
            echo "  notarytool: ** missing (needs Xcode 14 or later; altool is not accepted)"
        fi
    fi
    echo

    echo "== repository =="
    echo "  $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git repo>') on \
$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
    changed="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    echo "  uncommitted files: ${changed}"
} | tee "${report}"

echo
echo "==> written to ${report}"
