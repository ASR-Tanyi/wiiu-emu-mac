# Phase 1.5 — one warning set, applied to every first-party target.

set(WIIU_WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference
    -Wimplicit-fallthrough
    -Wunused
    -Wno-unknown-pragmas)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    list(APPEND WIIU_WARNINGS -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wuseless-cast)
endif()

# Apply the project warning set, and optionally sanitizers, to a target.
function(wiiu_set_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:${WIIU_WARNINGS}>)

    if(WIIU_WERROR)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Werror>)
    endif()

    if(WIIU_SANITIZE)
        # Not usable together with the JIT (Phase 12): ASan and MAP_JIT
        # regions do not coexist. Sanitized builds run the interpreter.
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${target} PRIVATE -fsanitize=address,undefined)
    endif()
endfunction()
