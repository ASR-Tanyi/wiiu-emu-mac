#!/usr/bin/env python3
"""Validate the disassembler against a real PowerPC assembler.

For every entry in the instruction table, ppcdis synthesises a legal encoding
and prints its disassembly. This script feeds that text back through
binutils' assembler and checks the result is bit-identical to the word it
came from.

That is a stronger check than comparing against objdump's text: it does not
care about spelling or operand-order conventions, only that the disassembly
*means* the same instruction. A wrong field position or a swapped operand
cannot survive it.

Usage:  ppc_roundtrip.py <path-to-ppcdis> [--as powerpc-linux-gnu-as]
Exit code is 0 only if every instruction round-trips.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# -many accepts the whole 750/Gekko set, paired singles included. -mbroadway
# is narrower and, oddly, assembles mftb as mfspr.
AS_FLAGS = ["-many"]


def assemble(assembler: str, objcopy: str, text: str, workdir: Path, index: int):
    """Assemble one instruction at address 0. Returns (word, error)."""
    source = workdir / f"i{index}.s"
    obj = workdir / f"i{index}.o"
    binary = workdir / f"i{index}.bin"
    source.write_text(f"\t.text\n\t{text}\n")

    result = subprocess.run(
        [assembler, *AS_FLAGS, "-o", str(obj), str(source)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, result.stderr.strip().splitlines()[-1] if result.stderr else "assembler failed"

    result = subprocess.run(
        [objcopy, "-O", "binary", "--only-section=.text", str(obj), str(binary)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, "objcopy failed"

    data = binary.read_bytes()
    if len(data) != 4:
        return None, f"produced {len(data)} bytes, expected 4"
    return int.from_bytes(data, "big"), None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ppcdis")
    parser.add_argument("--as", dest="assembler", default="powerpc-linux-gnu-as")
    parser.add_argument("--objcopy", dest="objcopy", default="powerpc-linux-gnu-objcopy")
    args = parser.parse_args()

    if shutil.which(args.assembler) is None:
        print(f"skipping: {args.assembler} not found", file=sys.stderr)
        return 77  # ctest treats this as "skipped"

    corpus = subprocess.run([args.ppcdis, "--corpus"], capture_output=True, text=True, check=True)
    entries = []
    for line in corpus.stdout.splitlines():
        word_hex, name, text = line.split("\t")
        entries.append((int(word_hex, 16), name, text))

    if not entries:
        print("corpus is empty", file=sys.stderr)
        return 1

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(assemble, args.assembler, args.objcopy, text, workdir, i)
                for i, (_, _, text) in enumerate(entries)
            ]
            results = [f.result() for f in futures]

    for (expected, name, text), (actual, error) in zip(entries, results):
        if error is not None:
            failures.append(f"{name:<12} {expected:08x}  {text!r}  -> {error}")
        elif actual != expected:
            failures.append(
                f"{name:<12} {expected:08x}  {text!r}  -> reassembled as {actual:08x}"
            )

    print(f"round-tripped {len(entries) - len(failures)}/{len(entries)} instructions")
    for failure in failures:
        print(f"  FAIL {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
