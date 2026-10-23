#!/usr/bin/env python3
"""Build RPX/RPL test files.

There is no Nintendo-produced RPX in this repository and there never will be,
so the parser is tested against files built here. The PowerPC code inside them
is assembled by binutils rather than hand-written, so the ELF structure and the
instruction stream are real; only the RPL-specific sections are constructed
from the documented layouts.

That is worth being precise about: this validates the parser against *this
script's* reading of the format. A file produced by Nintendo's own toolchain is
the only thing that validates the reading itself. Anyone with a legitimately
dumped RPX can point rpl-dump at it and compare.

Run after changing the fixtures, then commit the result:
    scripts/make_rpx.py
"""

import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "tests" / "loader" / "data"

SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_RELA = 4
SHT_STRTAB = 3
SHT_NOBITS = 8
SHT_RPL_EXPORTS = 0x80000001
SHT_RPL_IMPORTS = 0x80000002
SHT_RPL_CRCS = 0x80000003
SHT_RPL_FILEINFO = 0x80000004

SHF_WRITE = 0x1
SHF_ALLOC = 0x2
SHF_EXECINSTR = 0x4
SHF_DEFLATED = 0x08000000

ET_CAFE_RPL = 0xFE01  # what elf2rpl writes; see loader/rpl.hpp
EM_PPC = 20
RPL_IS_RPX = 0x2

TEXT_BASE = 0x02000000


def assemble(source: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "a.s").write_text("\t.text\n" + source)
        subprocess.run(["powerpc-linux-gnu-as", "-many", "-mregnames",
                        "-o", str(tmp / "a.o"), str(tmp / "a.s")],
                       check=True, capture_output=True, text=True)
        subprocess.run(["powerpc-linux-gnu-objcopy", "-O", "binary",
                        "--only-section=.text", str(tmp / "a.o"), str(tmp / "a.bin")],
                       check=True)
        return (tmp / "a.bin").read_bytes()


def read_object(path: Path):
    """Read a relocatable ELF32 big-endian object produced by the assembler.

    The point of going through a real object file rather than writing the
    tables by hand is that the symbols and relocations are then genuinely the
    ones a toolchain emits for `bl OSReport` and `lis r3,msg@ha` -- including
    the section symbols and addends that are easy to get wrong when inventing
    them.
    """
    blob = path.read_bytes()
    shoff, shentsize, shnum, shstrndx = struct.unpack_from(">I", blob, 32)[0], \
        struct.unpack_from(">H", blob, 46)[0], struct.unpack_from(">H", blob, 48)[0], \
        struct.unpack_from(">H", blob, 50)[0]

    headers = []
    for i in range(shnum):
        fields = struct.unpack_from(">IIIIIIIIII", blob, shoff + i * shentsize)
        headers.append(dict(zip(
            ["name", "type", "flags", "addr", "offset", "size",
             "link", "info", "align", "entsize"], fields)))

    def data(header):
        if header["type"] == SHT_NOBITS:
            return b""
        return blob[header["offset"]:header["offset"] + header["size"]]

    shstr = data(headers[shstrndx])

    def name_of(header):
        end = shstr.index(b"\0", header["name"])
        return shstr[header["name"]:end].decode()

    return blob, headers, data, name_of


def assemble_object(source: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "a.s").write_text(source)
    subprocess.run(["powerpc-linux-gnu-as", "-many", "-mregnames",
                    "-o", str(tmp / "a.o"), str(tmp / "a.s")],
                   check=True, capture_output=True, text=True)
    return tmp / "a.o"


class Section:
    def __init__(self, name, type_, flags=0, addr=0, data=b"",
                 link=0, info=0, align=1, entsize=0, compress=False):
        self.name = name
        self.type = type_
        self.flags = flags
        self.addr = addr
        self.data = data
        self.link = link
        self.info = info
        self.align = align
        self.entsize = entsize
        self.compress = compress
        self.offset = 0
        self.name_offset = 0

    size_override = 0

    def declared_size(self) -> int:
        return self.size_override or len(self.stored())

    def stored(self) -> bytes:
        """What actually lands in the file."""
        if not self.compress or not self.data:
            return self.data
        # A deflated section is a big-endian uncompressed length followed by a
        # zlib stream.
        return struct.pack(">I", len(self.data)) + zlib.compress(self.data, 6)

    def stored_flags(self) -> int:
        return self.flags | (SHF_DEFLATED if self.compress and self.data else 0)


def build_exports(entries, signature=0x12345678) -> bytes:
    """count, signature, then (value, name offset) pairs, then the names."""
    header = struct.pack(">II", len(entries), signature)
    table_size = 8 + len(entries) * 8

    names = bytearray()
    offsets = []
    for _, name in entries:
        offsets.append(table_size + len(names))
        names += name.encode() + b"\0"

    body = b"".join(struct.pack(">II", value, offset)
                    for (value, _), offset in zip(entries, offsets))
    return header + body + bytes(names)


def build_imports(library: str, count: int, signature=0x87654321) -> bytes:
    """count, signature, then the library name."""
    return struct.pack(">II", count, signature) + library.encode() + b"\0"


def build_file_info(*, text_size, data_size, load_size, flags, name: str,
                    stack_size=0x4000, heap_size=0x8000) -> bytes:
    """The 0x60-byte RplFileInfo, followed by its string table."""
    filename_offset = 0x60
    fields = [
        0xCAFE,          # version
        text_size,       # text_size
        32,              # text_align
        data_size,       # data_size
        32,              # data_align
        load_size,       # load_size
        32,              # load_align
        0,               # temp_size
        0,               # tramp_adjust
        0,               # sda_base
        0,               # sda2_base
        stack_size,      # stack_size
        filename_offset, # filename
        flags,           # flags
        heap_size,       # heap_size
        0,               # tag_offset
        0,               # min_version
        6,               # compression_level
        0,               # tramp_addition
        0,               # file_info_pad
        0x00021000,      # sdk_version
        0,               # sdk_revision
    ]
    blob = b"".join(struct.pack(">I", value) for value in fields)
    blob += struct.pack(">HH", 0, 0)        # tls_module_index, tls_align_shift
    blob += struct.pack(">I", 0)            # runtime_file_info_size
    assert len(blob) == 0x60, len(blob)
    return blob + name.encode() + b"\0"


def build(path: Path, sections, entry: int, *, note: str):
    # Section 0 is the reserved null entry.
    all_sections = [Section("", 0)] + sections

    # The name table has to exist before offsets can be assigned.
    shstrtab = bytearray(b"\0")
    for section in all_sections:
        if section.name:
            section.name_offset = len(shstrtab)
            shstrtab += section.name.encode() + b"\0"
        else:
            section.name_offset = 0
    name_section = Section(".shstrtab", SHT_STRTAB, data=bytes(shstrtab))
    name_section.name_offset = len(shstrtab)
    shstrtab += b".shstrtab\0"
    name_section.data = bytes(shstrtab)
    all_sections.append(name_section)

    # CRCs come next, one per section including the ones after it, so the
    # section has to be sized before contents are laid out.
    crc_index = len(all_sections)
    crc_section = Section(".rplcrcs", SHT_RPL_CRCS, entsize=4, align=4)
    crc_section.name_offset = len(shstrtab)
    shstrtab += b".rplcrcs\0"
    name_section.data = bytes(shstrtab)
    all_sections.append(crc_section)

    crcs = []
    for index, section in enumerate(all_sections):
        if index == crc_index or not section.data:
            crcs.append(0)
        else:
            # Over the uncompressed contents: the checksum describes the
            # section, not its storage.
            crcs.append(zlib.crc32(section.data) & 0xFFFFFFFF)
    crc_section.data = b"".join(struct.pack(">I", value) for value in crcs)

    header_size = 52
    entry_size = 40
    offset = header_size + entry_size * len(all_sections)

    for section in all_sections:
        stored = section.stored()
        if not stored:
            section.offset = 0
            continue
        align = max(section.align, 4)
        offset = (offset + align - 1) & ~(align - 1)
        section.offset = offset
        offset += len(stored)

    ident = bytes([0x7F, ord("E"), ord("L"), ord("F"), 1, 2, 1, 0xCA, 0xFE,
                   0, 0, 0, 0, 0, 0, 0])
    header = ident + struct.pack(
        ">HHIIIIIHHHHHH",
        ET_CAFE_RPL, EM_PPC, 1, entry, 0, header_size, 0,
        header_size, 0, 0, entry_size, len(all_sections), len(all_sections) - 2)

    out = bytearray(header)
    for section in all_sections:
        stored = section.stored()
        out += struct.pack(">IIIIIIIIII", section.name_offset, section.type,
                           section.stored_flags(), section.addr, section.offset,
                           section.declared_size(), section.link, section.info,
                           section.align, section.entsize)

    for section in all_sections:
        stored = section.stored()
        if not stored:
            continue
        out += b"\0" * (section.offset - len(out))
        out += stored

    path.write_bytes(bytes(out))
    print(f"  {path.name}: {len(out)} bytes, {len(all_sections)} sections -- {note}")


def build_homebrew(path: Path):
    """An RPX that calls into coreinit, with real symbols and relocations.

    This is what the HLE layer has to cope with: a call site relocated to an
    undefined symbol whose section index names the library it comes from, and
    a string address assembled from a lis/addi pair.
    """
    obj = assemble_object("""
\t.section .rodata
message:
\t.asciz "hello from wiiu-emu"
counter_msg:
\t.asciz "sum 1..10 = %d"

\t.text
\t.globl _start
_start:
\tmflr\t0
\tstwu\t1,-32(1)
\tstw\t0,36(1)

\tlis\t3,message@ha
\taddi\t3,3,message@l
\tbl\tOSReport

\tli\t5,0
\tli\t6,1
1:
\tcmpwi\t6,11
\tbeq\t2f
\tadd\t5,5,6
\taddi\t6,6,1
\tb\t1b
2:
\tlis\t3,counter_msg@ha
\taddi\t3,3,counter_msg@l
\tmr\t4,5
\tbl\tOSReport

\tlwz\t0,36(1)
\tmtlr\t0
\taddi\t1,1,32
\tli\t3,0
\tblr
""")

    _, headers, data, name_of = read_object(obj)

    by_name = {name_of(h): (i, h) for i, h in enumerate(headers) if name_of(h)}
    text_index, text_header = by_name[".text"]
    rodata_index, rodata_header = by_name[".rodata"]
    symtab_index, symtab_header = by_name[".symtab"]
    strtab_index, strtab_header = by_name[".strtab"]

    text = data(text_header)
    rodata = data(rodata_header)
    symtab = bytearray(data(symtab_header))
    strtab = data(strtab_header)

    # Where the sections will live in the RPX.
    text_addr = TEXT_BASE
    rodata_addr = 0x10000000

    # Section indices in the file we are about to write.
    new_text_index = 1
    new_rodata_index = 2
    new_import_index = 3

    def symbol_name(offset):
        end = strtab.index(b"\0", offset)
        return strtab[offset:end].decode()

    # Rewrite the symbol table for the new layout:
    #  - section symbols get the address their section will have, so a
    #    relocation against .rodata resolves to the real string;
    #  - undefined symbols are pointed at the .fimport_coreinit section, which
    #    is how an RPX says "this comes from coreinit".
    count = len(symtab) // 16
    for i in range(count):
        at = i * 16
        name_off, value, size, info, other, shndx = struct.unpack_from(">IIIBBH", symtab, at)
        sym_type = info & 0xF

        if shndx == text_index:
            shndx, value = new_text_index, value + text_addr
        elif shndx == rodata_index:
            shndx, value = new_rodata_index, value + rodata_addr
        elif shndx == 0 and name_off != 0:
            shndx = new_import_index          # undefined -> imported
        struct.pack_into(">IIIBBH", symtab, at, name_off, value, size, info, other, shndx)

    # Relocations: the assembler emitted them relative to the start of the
    # section, and they need to be addresses now.
    relocations = bytearray()
    for i, header in enumerate(headers):
        if header["type"] != SHT_RELA or header["info"] != text_index:
            continue
        raw = data(header)
        for j in range(len(raw) // 12):
            offset, info, addend = struct.unpack_from(">IIi", raw, j * 12)
            relocations += struct.pack(">IIi", offset + text_addr, info, addend)

    imports = build_imports("coreinit", 1)

    sections = [
        Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_addr, text, align=32),
        Section(".rodata", SHT_PROGBITS, SHF_ALLOC, rodata_addr, rodata, align=32),
        Section(".fimport_coreinit", SHT_RPL_IMPORTS, SHF_ALLOC, 0x10008000, imports, align=4),
        Section(".symtab", SHT_SYMTAB, 0, 0, bytes(symtab), link=5, entsize=16, align=4),
        Section(".strtab", 3, 0, 0, strtab, align=1),
        Section(".rela.text", SHT_RELA, 0, 0, bytes(relocations),
                link=4, info=new_text_index, entsize=12, align=4),
        Section(".rplfileinfo", SHT_RPL_FILEINFO, 0, 0,
                build_file_info(text_size=len(text), data_size=len(rodata),
                                load_size=0x1000, flags=RPL_IS_RPX,
                                name="homebrew.rpx"), align=4),
    ]

    # .symtab's link must point at .strtab, which is index 5 given the null
    # section at 0; likewise .rela.text links to .symtab at 4.
    build(path, sections, entry=text_addr,
          note=f"calls OSReport, {len(relocations) // 12} relocations")


def build_threaded(path: Path):
    """An RPX that creates a thread, waits for it, and reports from both.

    Exercises the whole path a real title takes: OSCreateThread with a stack
    the caller allocated, OSResumeThread to start it, OSJoinThread to wait, and
    a worker that returns a value through its exit code.
    """
    obj = assemble_object("""
\t.section .rodata
main_msg:
\t.asciz "main: starting worker"
worker_msg:
\t.asciz "worker: running"
joined_msg:
\t.asciz "main: worker returned %d"

\t.section .bss
\t.align 5
thread_obj:
\t.space 512
\t.align 5
thread_stack:
\t.space 4096
thread_stack_top:

\t.text
\t.globl _start
_start:
\tmflr\t0
\tstwu\t1,-32(1)
\tstw\t0,36(1)

\tlis\t3,main_msg@ha
\taddi\t3,3,main_msg@l
\tbl\tOSReport

\t# OSCreateThread(thread, entry, argc, argv, stackTop, stackSize, priority, attr)
\tlis\t3,thread_obj@ha
\taddi\t3,3,thread_obj@l
\tlis\t4,worker@ha
\taddi\t4,4,worker@l
\tli\t5,0
\tli\t6,0
\tlis\t7,thread_stack_top@ha
\taddi\t7,7,thread_stack_top@l
\tli\t8,4096
\tli\t9,16
\tli\t10,7
\tbl\tOSCreateThread

\tlis\t3,thread_obj@ha
\taddi\t3,3,thread_obj@l
\tbl\tOSResumeThread

\t# OSJoinThread(thread, &exitValue)
\tlis\t3,thread_obj@ha
\taddi\t3,3,thread_obj@l
\taddi\t4,1,8
\tbl\tOSJoinThread

\tlis\t3,joined_msg@ha
\taddi\t3,3,joined_msg@l
\tlwz\t4,8(1)
\tbl\tOSReport

\tlwz\t0,36(1)
\tmtlr\t0
\taddi\t1,1,32
\tli\t3,0
\tblr

worker:
\tmflr\t0
\tstwu\t1,-32(1)
\tstw\t0,36(1)

\tlis\t3,worker_msg@ha
\taddi\t3,3,worker_msg@l
\tbl\tOSReport

\tlwz\t0,36(1)
\tmtlr\t0
\taddi\t1,1,32
\tli\t3,99
\tblr
""")

    _, headers, data, name_of = read_object(obj)
    by_name = {name_of(h): (i, h) for i, h in enumerate(headers) if name_of(h)}

    text_index, text_header = by_name[".text"]
    rodata_index, rodata_header = by_name[".rodata"]
    bss_index, bss_header = by_name[".bss"]
    symtab_index, symtab_header = by_name[".symtab"]
    strtab_index, strtab_header = by_name[".strtab"]

    text = data(text_header)
    rodata = data(rodata_header)
    bss_size = bss_header["size"]
    symtab = bytearray(data(symtab_header))
    strtab = data(strtab_header)

    text_addr, rodata_addr, bss_addr = TEXT_BASE, 0x10000000, 0x10004000
    new_text, new_rodata, new_bss, new_import = 1, 2, 3, 4

    count = len(symtab) // 16
    for i in range(count):
        at = i * 16
        name_off, value, size, info, other, shndx = struct.unpack_from(">IIIBBH", symtab, at)
        if shndx == text_index:
            shndx, value = new_text, value + text_addr
        elif shndx == rodata_index:
            shndx, value = new_rodata, value + rodata_addr
        elif shndx == bss_index:
            shndx, value = new_bss, value + bss_addr
        elif shndx == 0 and name_off != 0:
            shndx = new_import
        struct.pack_into(">IIIBBH", symtab, at, name_off, value, size, info, other, shndx)

    relocations = bytearray()
    for header in headers:
        if header["type"] != SHT_RELA or header["info"] != text_index:
            continue
        raw = data(header)
        for j in range(len(raw) // 12):
            offset, info, addend = struct.unpack_from(">IIi", raw, j * 12)
            relocations += struct.pack(">IIi", offset + text_addr, info, addend)

    sections = [
        Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_addr, text, align=32),
        Section(".rodata", SHT_PROGBITS, SHF_ALLOC, rodata_addr, rodata, align=32),
        Section(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE, bss_addr, b"", align=32),
        Section(".fimport_coreinit", SHT_RPL_IMPORTS, SHF_ALLOC, 0x10008000,
                build_imports("coreinit", 4), align=4),
        Section(".symtab", SHT_SYMTAB, 0, 0, bytes(symtab), link=6, entsize=16, align=4),
        Section(".strtab", 3, 0, 0, strtab, align=1),
        Section(".rela.text", SHT_RELA, 0, 0, bytes(relocations),
                link=5, info=new_text, entsize=12, align=4),
        Section(".rplfileinfo", SHT_RPL_FILEINFO, 0, 0,
                build_file_info(text_size=len(text), data_size=len(rodata),
                                load_size=0x1000, flags=RPL_IS_RPX,
                                name="threaded.rpx"), align=4),
    ]
    # .bss is NOBITS but the loader must still zero it, so record its size.
    sections[2].size_override = bss_size

    build(path, sections, entry=text_addr,
          note=f"creates a thread, {len(relocations) // 12} relocations")


def build_clock(path: Path):
    """An RPX that asks the time and prints the date.

    The point is the argument convention as much as the clock: OSGetTime
    returns 64 bits in r3:r4, and OSTicksToCalendarTime takes that pair
    followed by a pointer, so the whole call lands correctly only if the
    64-bit pair rule is right. Nothing else in the fixtures exercises it.
    """
    obj = assemble_object("""
\t.section .rodata
date_msg:
\t.asciz "date %d-%d-%d %d:%d:%d"
round_msg:
\t.asciz "roundtrip %d"

\t.text
\t.globl _start
_start:
\tmflr\t0
\tstwu\t1,-96(1)
\tstw\t0,100(1)

\t# OSTicksToCalendarTime(OSGetTime(), &calendar) -- the pair stays in r3:r4
\tbl\tOSGetTime
\taddi\t5,1,16
\tbl\tOSTicksToCalendarTime

\tlis\t3,date_msg@ha
\taddi\t3,3,date_msg@l
\tlwz\t4,36(1)\t# year
\tlwz\t5,32(1)\t# month, 0-11
\taddi\t5,5,1
\tlwz\t6,28(1)\t# day
\tlwz\t7,24(1)\t# hour
\tlwz\t8,20(1)\t# minute
\tlwz\t9,16(1)\t# second
\tbl\tOSReport

\t# and back: the same calendar time must give the same tick count
\taddi\t3,1,16
\tbl\tOSCalendarTimeToTicks
\tmr\t9,4

\tlis\t3,round_msg@ha
\taddi\t3,3,round_msg@l
\tmr\t4,9
\tbl\tOSReport

\tlwz\t0,100(1)
\tmtlr\t0
\taddi\t1,1,96
\tli\t3,0
\tblr
""")

    _, headers, data, name_of = read_object(obj)
    by_name = {name_of(h): (i, h) for i, h in enumerate(headers) if name_of(h)}

    text_index, text_header = by_name[".text"]
    rodata_index, rodata_header = by_name[".rodata"]
    symtab_index, symtab_header = by_name[".symtab"]
    strtab_index, strtab_header = by_name[".strtab"]

    text = data(text_header)
    rodata = data(rodata_header)
    symtab = bytearray(data(symtab_header))
    strtab = data(strtab_header)

    text_addr, rodata_addr = TEXT_BASE, 0x10000000
    new_text, new_rodata, new_import = 1, 2, 3

    for i in range(len(symtab) // 16):
        at = i * 16
        name_off, value, size, info, other, shndx = struct.unpack_from(">IIIBBH", symtab, at)
        if shndx == text_index:
            shndx, value = new_text, value + text_addr
        elif shndx == rodata_index:
            shndx, value = new_rodata, value + rodata_addr
        elif shndx == 0 and name_off != 0:
            shndx = new_import
        struct.pack_into(">IIIBBH", symtab, at, name_off, value, size, info, other, shndx)

    relocations = bytearray()
    for header in headers:
        if header["type"] != SHT_RELA or header["info"] != text_index:
            continue
        raw = data(header)
        for j in range(len(raw) // 12):
            offset, info, addend = struct.unpack_from(">IIi", raw, j * 12)
            relocations += struct.pack(">IIi", offset + text_addr, info, addend)

    sections = [
        Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_addr, text, align=32),
        Section(".rodata", SHT_PROGBITS, SHF_ALLOC, rodata_addr, rodata, align=32),
        Section(".fimport_coreinit", SHT_RPL_IMPORTS, SHF_ALLOC, 0x10008000,
                build_imports("coreinit", 4), align=4),
        Section(".symtab", SHT_SYMTAB, 0, 0, bytes(symtab), link=5, entsize=16, align=4),
        Section(".strtab", 3, 0, 0, strtab, align=1),
        Section(".rela.text", SHT_RELA, 0, 0, bytes(relocations),
                link=4, info=new_text, entsize=12, align=4),
        Section(".rplfileinfo", SHT_RPL_FILEINFO, 0, 0,
                build_file_info(text_size=len(text), data_size=len(rodata),
                                load_size=0x1000, flags=RPL_IS_RPX,
                                name="clock.rpx"), align=4),
    ]

    build(path, sections, entry=text_addr,
          note=f"asks the time, {len(relocations) // 12} relocations")


def build_dataimport(path: Path):
    """An RPX laid out as wut's elf2rpl lays one out: its import table in the
    load region at 0xC0000000, which Cafe OS's loader keeps for itself, and a
    call through coreinit's MEMAllocFromDefaultHeap -- which coreinit exports
    as data, a pointer to the default heap's allocator, so the code loads the
    pointer and calls through it.
    """
    obj = assemble_object("""
\t.text
\t.globl _start
_start:
\tmflr\t0
\tstwu\t1,-32(1)
\tstw\t0,36(1)

\tlis\t9,MEMAllocFromDefaultHeap@ha
\tlwz\t9,MEMAllocFromDefaultHeap@l(9)
\tmtctr\t9
\tli\t3,64
\tbctrl

\tlwz\t0,36(1)
\tmtlr\t0
\taddi\t1,1,32
\tblr
""")

    _, headers, data, name_of = read_object(obj)
    by_name = {name_of(h): (i, h) for i, h in enumerate(headers) if name_of(h)}
    text_index, text_header = by_name[".text"]
    _, symtab_header = by_name[".symtab"]
    _, strtab_header = by_name[".strtab"]

    text = data(text_header)
    symtab = bytearray(data(symtab_header))
    strtab = data(strtab_header)

    text_addr = TEXT_BASE
    load_addr = 0xC0000000      # wut.ld's loadmem
    new_text_index = 1
    new_import_index = 2

    for i in range(len(symtab) // 16):
        at = i * 16
        name_off, value, size, info, other, shndx = struct.unpack_from(">IIIBBH", symtab, at)
        if shndx == text_index:
            shndx, value = new_text_index, value + text_addr
        elif shndx == 0 and name_off != 0:
            # The data import: a symbol in .dimport_coreinit, past the
            # table's header and the library's name.
            shndx, value = new_import_index, load_addr + 0x18
        struct.pack_into(">IIIBBH", symtab, at, name_off, value, size, info, other, shndx)

    relocations = bytearray()
    for header in headers:
        if header["type"] != SHT_RELA or header["info"] != text_index:
            continue
        raw = data(header)
        for j in range(len(raw) // 12):
            offset, info, addend = struct.unpack_from(">IIi", raw, j * 12)
            relocations += struct.pack(">IIi", offset + text_addr, info, addend)

    imports = build_imports("coreinit", 1) + bytes(0x18 - 8 - len(b"coreinit\0")) + bytes(8)

    sections = [
        Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_addr, text, align=32),
        Section(".dimport_coreinit", SHT_RPL_IMPORTS, SHF_ALLOC, load_addr, imports, align=16),
        Section(".symtab", SHT_SYMTAB, 0, 0, bytes(symtab), link=4, entsize=16, align=4),
        Section(".strtab", 3, 0, 0, strtab, align=1),
        Section(".rela.text", SHT_RELA, 0, 0, bytes(relocations),
                link=3, info=new_text_index, entsize=12, align=4),
        Section(".rplfileinfo", SHT_RPL_FILEINFO, 0, 0,
                build_file_info(text_size=len(text), data_size=0,
                                load_size=len(imports), flags=RPL_IS_RPX,
                                name="dataimport.rpx"), align=4),
    ]
    build(path, sections, entry=text_addr,
          note="imports in the load region, a data import called through")


def main() -> int:
    if shutil.which("powerpc-linux-gnu-as") is None:
        print("powerpc-linux-gnu-as not found; cannot rebuild fixtures", file=sys.stderr)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- a compressed executable ------------------------------------------
    text = assemble("""
        mflr    0
        stwu    1,-32(1)
        stw     0,36(1)
        li      3,0
        li      4,1
    loop:
        cmpwi   4,11
        beq     done
        add     3,3,4
        addi    4,4,1
        b       loop
    done:
        lwz     0,36(1)
        mtlr    0
        addi    1,1,32
        blr
    """)
    rodata = b"hello from a test rpx\0"
    data = struct.pack(">IIII", 1, 2, 3, 4)

    build(
        OUTPUT_DIR / "hello.rpx",
        [
            Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                    TEXT_BASE, text, align=32, compress=True),
            Section(".rodata", SHT_PROGBITS, SHF_ALLOC,
                    0x10000000, rodata, align=32, compress=True),
            Section(".data", SHT_PROGBITS, SHF_ALLOC | SHF_WRITE,
                    0x10001000, data, align=32),
            Section(".bss", SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0x10002000, b"", align=32),
            Section(".fimport_coreinit", SHT_RPL_IMPORTS, SHF_ALLOC,
                    0x10003000, build_imports("coreinit", 3), align=4),
            Section(".fimport_gx2", SHT_RPL_IMPORTS, SHF_ALLOC,
                    0x10003100, build_imports("gx2", 1), align=4),
            Section(".rplfileinfo", SHT_RPL_FILEINFO, 0, 0,
                    build_file_info(text_size=len(text), data_size=len(data),
                                    load_size=0x1000, flags=RPL_IS_RPX,
                                    name="hello.rpx"), align=4),
        ],
        entry=TEXT_BASE,
        note="compressed, RPL_IS_RPX set",
    )

    # --- an uncompressed library with exports ------------------------------
    lib_text = assemble("""
        add     3,3,4
        blr
        subf    3,4,3
        blr
    """)
    build(
        OUTPUT_DIR / "libtest.rpl",
        [
            Section(".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
                    TEXT_BASE, lib_text, align=32),
            Section(".fexports", SHT_RPL_EXPORTS, SHF_ALLOC, 0x10000000,
                    build_exports([(TEXT_BASE, "TestAdd"),
                                   (TEXT_BASE + 8, "TestSubtract")]), align=4),
            Section(".dexports", SHT_RPL_EXPORTS, SHF_ALLOC, 0x10000100,
                    build_exports([(0x10001000, "g_TestCounter")]), align=4),
            Section(".rplfileinfo", SHT_RPL_FILEINFO, 0, 0,
                    build_file_info(text_size=len(lib_text), data_size=0,
                                    load_size=0x100, flags=0,
                                    name="libtest.rpl"), align=4),
        ],
        entry=0,
        note="uncompressed library, exports present",
    )

    build_homebrew(OUTPUT_DIR / "homebrew.rpx")
    build_threaded(OUTPUT_DIR / "threaded.rpx")
    build_clock(OUTPUT_DIR / "clock.rpx")
    build_dataimport(OUTPUT_DIR / "dataimport.rpx")

    return 0


if __name__ == "__main__":
    sys.exit(main())
