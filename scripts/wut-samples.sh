#!/usr/bin/env bash
# Build wut's `helloworld` sample into tests/loader/data/wut/helloworld.rpx.
#
# A real-toolchain RPX: compiled against wut and newlib and turned into an RPX
# by wut-tools' elf2rpl, which is what the loader has to read in the end. Like
# every RPX here it is built, never committed (*.rpx is ignored), and the tests
# that want it skip when it is not there.
#
# Two ways to get one:
#
#   * With devkitPro installed ($DEVKITPRO/wut exists -- `sudo dkp-pacman -S
#     wiiu-dev` on a Mac), the sample is built with devkitPPC, as anyone else
#     would build it.
#
#   * Without it, on Debian or Ubuntu, a stand-in for devkitPPC is put
#     together under build/wut/: Ubuntu's powerpc-linux-gnu GCC, run as an
#     EABI compiler for the 750 (-meabi -mlong-double-64, no Linux macros, no
#     PIE, none of glibc's headers); devkitPro's own newlib, built with it;
#     elf2rpl and rplimportgen, built for the host; and the parts of libwut
#     the sample can pull in -- the import stubs, the crt, the newlib glue,
#     malloc, the FSA devoptab and WHB. Two things devkitPPC supplies and the
#     Linux compiler does not are written here: GCC's EABI .init/.fini
#     prologue and epilogue (ecrti/ecrtn), and a <mutex> and <type_traits>
#     with just what wut's C++ uses. Everything is fetched from GitHub with
#     git, at pinned commits, so this works where only git and apt reach
#     (devkitPro's package servers do not have to). Needs:
#       apt-get install gcc-powerpc-linux-gnu g++-powerpc-linux-gnu \
#                       binutils-powerpc-linux-gnu zlib1g-dev texinfo git make
#
# Usage: scripts/wut-samples.sh      (from the repository root or anywhere)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$ROOT/build/wut"
OUT="$ROOT/tests/loader/data/wut"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

# What was built from, by commit.
WUT_COMMIT=937f0d6901f5e2a72672154a674ec47b63e2f417        # devkitPro/wut
NEWLIB_COMMIT=57bd8e1ade3cf46d2e61a72516ac8bfaf8fdfd62     # devkitPro/newlib, branch devkitPro
WUT_TOOLS_COMMIT=b22de1a3c0c8950c4ad4b4df170c89c05d3196aa  # devkitPro/wut-tools

# fetch NAME URL COMMIT: a shallow clone at exactly that commit.
fetch() {
    local dir="$WORK/src/$1"
    if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$3" ]; then
        return
    fi
    rm -rf "$dir"
    git init -q "$dir"
    git -C "$dir" remote add origin "$2"
    git -C "$dir" fetch -q --depth 1 origin "$3"
    git -C "$dir" checkout -q FETCH_HEAD
}

mkdir -p "$WORK/src" "$OUT"
fetch wut https://github.com/devkitPro/wut.git "$WUT_COMMIT"
SAMPLE="$WORK/src/wut/samples/make/helloworld"

# --- with devkitPro --------------------------------------------------------
if [ -n "${DEVKITPRO:-}" ] && [ -f "$DEVKITPRO/wut/share/wut_rules" ]; then
    echo "devkitPro at $DEVKITPRO: building with devkitPPC"
    export DEVKITPPC="${DEVKITPPC:-$DEVKITPRO/devkitPPC}"
    make -C "$SAMPLE" -j"$JOBS"
    cp "$SAMPLE/helloworld.rpx" "$OUT/helloworld.rpx"
    echo "wrote $OUT/helloworld.rpx"
    exit 0
fi

# --- without: the stand-in -------------------------------------------------
for tool in powerpc-linux-gnu-gcc powerpc-linux-gnu-g++ powerpc-linux-gnu-ld git make; do
    if ! command -v "$tool" >/dev/null; then
        echo "no devkitPro, and $tool is missing for the stand-in toolchain; see the top of $0" >&2
        exit 1
    fi
done
echo "no devkitPro: building a stand-in for devkitPPC under $WORK"

DKP="$WORK/dkp"
BIN="$DKP/bin"
mkdir -p "$BIN" "$DKP/shim" "$DKP/lib"
GCCINC="$(powerpc-linux-gnu-gcc -print-file-name=include)"
COMMON="-mcpu=750 -meabi -mhard-float -mlong-double-64 -fno-pie -fno-pic -fno-stack-protector \
  -U__linux__ -U__linux -Ulinux -U__gnu_linux__ -U__unix__ -U__unix -Uunix \
  -D__DEVKITPPC__ -D__DEVKITPRO__"

cat > "$BIN/powerpc-eabi-gcc" <<EOF
#!/bin/sh
# Ubuntu's powerpc-linux-gnu GCC standing in for devkitPPC's powerpc-eabi-gcc.
exec powerpc-linux-gnu-gcc $COMMON \\
  -nostdinc -isystem "$DKP/powerpc-eabi/include" -isystem "$GCCINC" "\$@"
EOF
cat > "$BIN/powerpc-eabi-g++" <<EOF
#!/bin/sh
# The same for C++, with no libstdc++ but the shims beside this script.
exec powerpc-linux-gnu-g++ $COMMON -fno-exceptions -fno-rtti \\
  -nostdinc -nostdinc++ -isystem "$DKP/shim" -isystem "$DKP/powerpc-eabi/include" \\
  -isystem "$GCCINC" "\$@"
EOF
chmod +x "$BIN/powerpc-eabi-gcc" "$BIN/powerpc-eabi-g++"
for tool in ar as ld nm ranlib objcopy readelf strip; do
    ln -sf "$(command -v powerpc-linux-gnu-$tool)" "$BIN/powerpc-eabi-$tool"
done
export PATH="$BIN:$PATH"

# The C++ wut's devoptab uses: std::scoped_lock over one lockable, and
# std::underlying_type_t for its enum operators.
cat > "$DKP/shim/mutex" <<'EOF'
#pragma once
namespace std {
template <typename... M> class scoped_lock;
template <typename M> class scoped_lock<M> {
public:
    explicit scoped_lock(M& m) : m_(m) { m_.lock(); }
    ~scoped_lock() { m_.unlock(); }
    scoped_lock(const scoped_lock&) = delete;
    scoped_lock& operator=(const scoped_lock&) = delete;
private:
    M& m_;
};
template <typename M> scoped_lock(M&) -> scoped_lock<M>;
}
EOF
cat > "$DKP/shim/type_traits" <<'EOF'
#pragma once
namespace std {
template <typename T> struct underlying_type { using type = __underlying_type(T); };
template <typename T> using underlying_type_t = typename underlying_type<T>::type;
}
EOF
for pair in cstdio:stdio.h cstring:string.h cstdlib:stdlib.h cerrno:errno.h \
            cstdint:stdint.h cstddef:stddef.h climits:limits.h; do
    printf '#pragma once\n#include <%s>\n' "${pair##*:}" > "$DKP/shim/${pair%%:*}"
done

# GCC's EABI .init/.fini prologue and epilogue (libgcc's eabi-ci.S and
# eabi-cn.S), which newlib's __libc_fini_array reaches through __fini.
cat > "$DKP/lib/ecrti.S" <<'EOF'
	.section ".init","ax"
	.align 2
	.globl __init
	.type __init,@function
__init:
	stwu 1,-16(1)
	mflr 0
	stw 0,20(1)
	.section ".fini","ax"
	.align 2
	.globl __fini
	.type __fini,@function
__fini:
	stwu 1,-16(1)
	mflr 0
	stw 0,20(1)
	.section .note.GNU-stack,"",@progbits
EOF
cat > "$DKP/lib/ecrtn.S" <<'EOF'
	.section ".init","ax"
	lwz 0,20(1)
	mtlr 0
	addi 1,1,16
	blr
	.section ".fini","ax"
	lwz 0,20(1)
	mtlr 0
	addi 1,1,16
	blr
	.section .note.GNU-stack,"",@progbits
EOF
powerpc-eabi-gcc -c "$DKP/lib/ecrti.S" -o "$DKP/lib/ecrti.o"
powerpc-eabi-gcc -c "$DKP/lib/ecrtn.S" -o "$DKP/lib/ecrtn.o"

# devkitPro's newlib, as devkitPro's buildscripts configure it for
# powerpc-eabi -- with no multilibs, which the Linux compiler would otherwise
# ask for.
if [ ! -f "$DKP/powerpc-eabi/lib/libc.a" ]; then
    fetch newlib https://github.com/devkitPro/newlib.git "$NEWLIB_COMMIT"
    rm -rf "$WORK/newlib-build"
    mkdir -p "$WORK/newlib-build"
    (
        cd "$WORK/newlib-build"
        CFLAGS_FOR_TARGET="-O2 -ffunction-sections -fdata-sections -DCUSTOM_MALLOC_LOCK" \
        CC_FOR_TARGET=powerpc-eabi-gcc AR_FOR_TARGET=powerpc-eabi-ar \
        AS_FOR_TARGET=powerpc-eabi-as LD_FOR_TARGET=powerpc-eabi-ld \
        NM_FOR_TARGET=powerpc-eabi-nm RANLIB_FOR_TARGET=powerpc-eabi-ranlib \
        OBJCOPY_FOR_TARGET=powerpc-eabi-objcopy READELF_FOR_TARGET=powerpc-eabi-readelf \
        "$WORK/src/newlib/configure" --target=powerpc-eabi --prefix="$DKP" \
            --disable-multilib --disable-newlib-supplied-syscalls --enable-newlib-mb \
            --disable-newlib-wide-orient --enable-newlib-register-fini > configure.log
        make -j"$JOBS" > make.log
        make install -j1 > install.log
    )
fi

# elf2rpl and rplimportgen, for the host.
if [ ! -x "$BIN/elf2rpl" ]; then
    fetch wut-tools https://github.com/devkitPro/wut-tools.git "$WUT_TOOLS_COMMIT"
    git -C "$WORK/src/wut-tools" submodule update -q --init --depth 1
    T="$WORK/src/wut-tools"
    flags=(-std=c++17 -O1 -I"$T/src/common" -I"$T/libraries/excmd/src" -I"$T/libraries/fmt/include")
    g++ "${flags[@]}" "$T/src/elf2rpl/main.cpp" "$T/libraries/fmt/src/format.cc" -lz -o "$BIN/elf2rpl"
    g++ "${flags[@]}" "$T"/src/rplimportgen/*.cpp "$T/libraries/fmt/src/format.cc" -lz \
        -o "$BIN/rplimportgen"
fi

# libwut: what helloworld can pull in.
W="$WORK/src/wut"
OBJ="$WORK/libwut"
rm -rf "$OBJ"
mkdir -p "$OBJ"
INC=(-I"$W/include" -I"$W/libraries/libwhb/include" -I"$W/libraries/libgfd/include")
CFLAGS=(-O2 -g -Wall -ffunction-sections -fdata-sections -DESPRESSO -D__WIIU__ -D__WUT__ "${INC[@]}")
for def in "$W"/cafe/*.def; do
    name="$(basename "$def" .def)"
    rplimportgen "$def" "$OBJ/$name.s" > /dev/null
    powerpc-eabi-gcc -x assembler-with-cpp -c "$OBJ/$name.s" -o "$OBJ/def_$name.o"
done
for s in "$W"/libraries/wutcrt/*.s; do
    powerpc-eabi-gcc -x assembler-with-cpp -c "$s" -o "$OBJ/crt_$(basename "$s" .s).o"
done
for c in "$W"/libraries/wutcrt/*.c "$W"/libraries/wutnewlib/*.c "$W"/libraries/wutmalloc/*.c \
         "$W"/libraries/wutdefaultheap/*.c "$W"/libraries/libwhb/src/*.c \
         "$W"/libraries/libgfd/src/*.c; do
    powerpc-eabi-gcc "${CFLAGS[@]}" -c "$c" \
        -o "$OBJ/$(basename "$(dirname "$c")")_$(basename "$c" .c).o"
done
for cpp in "$W"/libraries/wutdevoptab/*.cpp; do
    powerpc-eabi-g++ "${CFLAGS[@]}" -std=gnu++17 -c "$cpp" \
        -o "$OBJ/devoptab_$(basename "$cpp" .cpp).o"
done
rm -f "$OBJ/libwut.a"
powerpc-eabi-ar rcs "$OBJ/libwut.a" "$OBJ"/*.o

# The sample, linked by wut's own script as wut.specs links it.
powerpc-eabi-gcc "${CFLAGS[@]}" -c "$SAMPLE/source/main.c" -o "$WORK/helloworld.o"
powerpc-eabi-ld -T "$W/share/wut.ld" --gc-sections --emit-relocs --no-eh-frame-hdr \
    -z nocopyreloc -static -z noexecstack \
    "$DKP/lib/ecrti.o" "$WORK/helloworld.o" -L"$OBJ" -L"$DKP/powerpc-eabi/lib" \
    --start-group -lwut -lc -lm -lsysbase "$(powerpc-linux-gnu-gcc -print-libgcc-file-name)" \
    --end-group "$DKP/lib/ecrtn.o" -o "$WORK/helloworld.elf"
# The Linux toolchain marks every object's stack; elf2rpl knows no such section.
powerpc-eabi-objcopy --remove-section=.note.GNU-stack "$WORK/helloworld.elf"
elf2rpl "$WORK/helloworld.elf" "$OUT/helloworld.rpx"
echo "wrote $OUT/helloworld.rpx"
