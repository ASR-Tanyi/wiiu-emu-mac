#!/usr/bin/env python3
"""Generate test fixtures for the PowerPC interpreter.

Two kinds, both written into tests/cpu/fixtures.inc and committed so the tests
run anywhere:

1. Golden vectors computed by an *independent* implementation in this file.
   Re-deriving rotate masks and carry/overflow rules from the architecture
   description in a different language is a real second opinion; checking the
   C++ against itself would not be.

2. Real programs assembled by binutils. Hand-written machine code would only
   prove the interpreter agrees with my arithmetic, not with a toolchain.

Run after changing the fixtures, then commit the result:
    scripts/gen_cpu_fixtures.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "tests" / "cpu" / "fixtures.inc"

MASK32 = 0xFFFFFFFF


# --- independent reference implementations ---------------------------------

def rotate_mask(mb: int, me: int) -> int:
    """The mask PowerPC builds from MB and ME.

    Derived from the definition directly: bits MB through ME inclusive are set,
    counting from the most significant end, wrapping when MB > ME.
    """
    mask = 0
    bit = mb
    while True:
        mask |= 1 << (31 - bit)
        if bit == me:
            break
        bit = (bit + 1) % 32
    return mask


def rotl32(value: int, amount: int) -> int:
    amount &= 31
    return ((value << amount) | (value >> (32 - amount))) & MASK32 if amount else value


def add_carry(a: int, b: int, carry_in: int):
    wide = a + b + carry_in
    return wide & MASK32, 1 if wide > MASK32 else 0


def signed(value: int) -> int:
    return value - 0x100000000 if value & 0x80000000 else value


def add_overflow(a: int, b: int) -> int:
    """Signed overflow: operands share a sign and the result differs."""
    result = (a + b) & MASK32
    return 1 if (~(a ^ b) & (a ^ result) & 0x80000000) else 0


def srawi_carry(value: int, shift: int) -> int:
    """Carry is set when a negative value loses one bits off the bottom."""
    if shift == 0:
        return 0
    lost = value & ((1 << shift) - 1)
    return 1 if signed(value) < 0 and lost else 0


# --- assembled programs ----------------------------------------------------

# Single instructions the tests execute directly. Hand-writing these hex words
# is exactly the mistake the round-trip check exists to eliminate -- two of
# them were wrong the first time -- so they come from the assembler too.
ENCODINGS = {
    "addc": "addc 3,4,5",
    "adde": "adde 3,4,5",
    "addo": "addo 3,4,5",
    "subf": "subf 3,4,5",
    "subf_rc": "subf. 3,4,5",
    "divw": "divw 3,4,5",
    "srawi_base": "srawi 4,3,0",
    "slw": "slw 4,3,5",
    "srw": "srw 4,3,5",
    "li_r3_5": "li 3,5",
    "cmp": "cmp 0,0,3,4",
    "cmpl": "cmpl 0,0,3,4",
    "stw_16_r4": "stw 3,16(4)",
    "lwz_16_r4": "lwz 3,16(4)",
    "lhz_16_r4": "lhz 3,16(4)",
    "lbz_16_r4": "lbz 3,16(4)",
    "lha_0_r4": "lha 3,0(4)",
    "lhz_0_r4": "lhz 3,0(4)",
    "lwbrx": "lwbrx 3,4,5",
    "stwu_neg32": "stwu 3,-32(1)",
    "lwz_0_r4": "lwz 3,0(4)",
    "stmw_r28": "stmw 28,0(4)",
    "lmw_r28": "lmw 28,0(4)",
    "lwarx": "lwarx 3,4,5",
    "stwcx": "stwcx. 3,4,5",
    "dcbf": "dcbf 4,5",
    "bl_0x100": "bl 0x100",
    "blr": "blr",
    "beq_0x20": "beq 0x20",
    "bdnz_back4": "bdnz . - 4",
    "mtlr_r3": "mtlr 3",
    "mflr_r4": "mflr 4",
    "mtctr_r3": "mtctr 3",
    "sc": "sc",
    "fadds": "fadds 1,2,3",
    "fadd": "fadd 1,2,3",
    "fsub": "fsub 1,2,3",
    "fmul": "fmul 1,2,4",
    "fmuls": "fmuls 1,2,4",
    "fdiv": "fdiv 1,2,3",
    "fmadd": "fmadd 1,2,4,3",
    "fmadds": "fmadds 1,2,4,3",
    "fmsub": "fmsub 1,2,4,3",
    "fnmadd": "fnmadd 1,2,4,3",
    "fsel": "fsel 1,2,4,3",
    "fmr": "fmr 1,3",
    "fneg": "fneg 1,3",
    "fabs": "fabs 1,3",
    "frsp": "frsp 1,3",
    "fctiwz": "fctiwz 1,3",
    "fcmpu": "fcmpu 0,2,3",
    "fcmpo": "fcmpo 0,2,3",
    "lfs_0_r4": "lfs 1,0(4)",
    "lfd_0_r4": "lfd 1,0(4)",
    "stfs_0_r4": "stfs 1,0(4)",
    "stfd_0_r4": "stfd 1,0(4)",
    "ps_add": "ps_add 1,2,3",
    "ps_sub": "ps_sub 1,2,3",
    "ps_mul": "ps_mul 1,2,4",
    "ps_muls0": "ps_muls0 1,2,4",
    "ps_muls1": "ps_muls1 1,2,4",
    "ps_madd": "ps_madd 1,2,4,3",
    "ps_sel": "ps_sel 1,2,4,3",
    "ps_neg": "ps_neg 1,3",
    "ps_merge00": "ps_merge00 1,2,3",
    "ps_merge01": "ps_merge01 1,2,3",
    "ps_merge10": "ps_merge10 1,2,3",
    "ps_merge11": "ps_merge11 1,2,3",
    "psq_l_gqr2": "psq_l 1,0(4),0,2",
    "psq_l_w1_gqr2": "psq_l 1,0(4),1,2",
    "psq_st_gqr2": "psq_st 1,0(4),0,2",
    "mtspr_gqr2": "mtspr 914,3",
}

PROGRAMS = [
    ("sum_to_ten", """
        li      3,0
        li      4,1
        li      5,11
    loop:
        cmpw    4,5
        beq     done
        add     3,3,4
        addi    4,4,1
        b       loop
    done:
        blr
    """),
    ("countdown_ctr", """
        li      3,0
        li      4,10
        mtctr   4
    loop:
        addi    3,3,2
        bdnz    loop
        blr
    """),
    ("copy_bytes", """
        li      5,8
        mtctr   5
    loop:
        lbz     6,0(3)
        stb     6,0(4)
        addi    3,3,1
        addi    4,4,1
        bdnz    loop
        blr
    """),
    ("add64", """
        addc    4,4,6
        adde    3,3,5
        blr
    """),
    ("call_and_return", """
        mflr    0
        stwu    1,-16(1)
        stw     0,20(1)
        bl      callee
        lwz     0,20(1)
        mtlr    0
        addi    1,1,16
        blr
    callee:
        addi    3,3,42
        blr
    """),
    ("dot_product", """
        psq_l   1,0(3),0,0
        psq_l   2,8(3),0,0
        psq_l   3,0(4),0,0
        psq_l   4,8(4),0,0
        ps_mul  5,1,3
        ps_madd 5,2,4,5
        ps_add  6,5,5
        blr
    """),
    ("float_average", """
        lfs     1,0(3)
        lfs     2,4(3)
        fadds   3,1,2
        lfs     4,8(3)
        fdivs   3,3,4
        stfs    3,0(4)
        blr
    """),
    ("bitfield_ops", """
        rlwinm  4,3,8,8,20
        srawi   5,3,4
        extsb   6,3
        cntlzw  7,3
        blr
    """),
]


def assemble(source: str, assembler: str, objcopy: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "p.s").write_text("\t.text\n" + source)
        subprocess.run([assembler, "-many", "-mregnames", "-o", str(tmp / "p.o"), str(tmp / "p.s")],
                       check=True, capture_output=True, text=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text",
                        str(tmp / "p.o"), str(tmp / "p.bin")], check=True)
        return (tmp / "p.bin").read_bytes()


def main() -> int:
    assembler = "powerpc-linux-gnu-as"
    objcopy = "powerpc-linux-gnu-objcopy"
    if shutil.which(assembler) is None:
        print(f"{assembler} not found; cannot regenerate fixtures", file=sys.stderr)
        return 1

    out = []
    out.append("// Generated by scripts/gen_cpu_fixtures.py -- do not edit.")
    out.append("//")
    out.append("// Golden vectors come from an independent implementation in that script;")
    out.append("// the programs are assembled by binutils' PowerPC assembler.")
    out.append("")

    # Rotate masks: every MB/ME pair, including the MB > ME wrapping cases that
    # are easy to get wrong and produce plausible-looking garbage.
    out.append("constexpr std::array<u32, 1024> kRotateMasks{{")
    row = []
    for mb in range(32):
        for me in range(32):
            row.append(f"0x{rotate_mask(mb, me):08X}U")
    for i in range(0, len(row), 6):
        out.append("    " + ", ".join(row[i:i + 6]) + ",")
    out.append("}};")
    out.append("")

    # Carry and overflow over values chosen to sit on the boundaries.
    interesting = [0x00000000, 0x00000001, 0x0000FFFF, 0x7FFFFFFE, 0x7FFFFFFF,
                   0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF, 0x12345678]
    out.append("struct AddVector {")
    out.append("    u32 a;")
    out.append("    u32 b;")
    out.append("    u32 carry_in;")
    out.append("    u32 sum;")
    out.append("    u32 carry_out;")
    out.append("    u32 overflow;")
    out.append("};")
    out.append("")
    vectors = []
    for a in interesting:
        for b in interesting:
            for carry_in in (0, 1):
                total, carry_out = add_carry(a, b, carry_in)
                vectors.append(
                    f"    {{0x{a:08X}U, 0x{b:08X}U, {carry_in}U, 0x{total:08X}U, "
                    f"{carry_out}U, {add_overflow(a, b)}U}},")
    out.append(f"constexpr std::array<AddVector, {len(vectors)}> kAddVectors{{{{")
    out.extend(vectors)
    out.append("}};")
    out.append("")

    # srawi: result and carry for every shift amount.
    out.append("struct ShiftVector {")
    out.append("    u32 value;")
    out.append("    u32 shift;")
    out.append("    u32 result;")
    out.append("    u32 carry;")
    out.append("};")
    out.append("")
    shifts = []
    for value in interesting:
        for shift in range(32):
            result = (signed(value) >> shift) & MASK32
            shifts.append(f"    {{0x{value:08X}U, {shift}U, 0x{result:08X}U, "
                          f"{srawi_carry(value, shift)}U}},")
    out.append(f"constexpr std::array<ShiftVector, {len(shifts)}> kShiftVectors{{{{")
    out.extend(shifts)
    out.append("}};")
    out.append("")

    # Single instructions, so no test has to hand-write a hex word.
    out.append("struct NamedEncoding {")
    out.append("    std::string_view name;")
    out.append("    u32 word;")
    out.append("};")
    out.append("")
    for name, source in ENCODINGS.items():
        code = assemble("\t" + source + "\n", assembler, objcopy)
        word = int.from_bytes(code[:4], "big")
        out.append(f"constexpr u32 kEnc_{name} = 0x{word:08X}U;  // {source}")
    out.append("")

    # Assembled programs.
    for name, source in PROGRAMS:
        code = assemble(source, assembler, objcopy)
        words = [f"0x{int.from_bytes(code[i:i + 4], 'big'):08X}U"
                 for i in range(0, len(code), 4)]
        out.append(f"// {name}: {len(words)} instructions")
        out.append(f"constexpr std::array<u32, {len(words)}> kProgram_{name}{{{{")
        for i in range(0, len(words), 4):
            out.append("    " + ", ".join(words[i:i + 4]) + ",")
        out.append("}};")
        out.append("")

    OUTPUT.write_text("\n".join(out) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)} "
          f"({len(vectors)} add vectors, {len(shifts)} shift vectors, "
          f"{len(PROGRAMS)} programs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
