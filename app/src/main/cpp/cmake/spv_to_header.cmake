# spv_to_header.cmake
# Converts a SPIR-V binary file into a C++ header with a uint32_t array.
#
# Usage: cmake -DSPV_FILE=input.spv -DHEADER_FILE=output.h -DVAR_NAME=shader_data -P spv_to_header.cmake

if(NOT DEFINED SPV_FILE OR NOT DEFINED HEADER_FILE OR NOT DEFINED VAR_NAME)
    message(FATAL_ERROR "Usage: cmake -DSPV_FILE=<path> -DHEADER_FILE=<path> -DVAR_NAME=<name> -P spv_to_header.cmake")
endif()

file(READ "${SPV_FILE}" SPV_HEX HEX)
string(LENGTH "${SPV_HEX}" SPV_HEX_LEN)

# Each byte is 2 hex chars; SPIR-V is uint32_t aligned (4 bytes = 8 hex chars per word)
math(EXPR WORD_COUNT "${SPV_HEX_LEN} / 8")

set(BODY "")
set(COL 0)
math(EXPR LAST_WORD "${WORD_COUNT} - 1")

foreach(I RANGE ${LAST_WORD})
    math(EXPR OFFSET "${I} * 8")

    # Read 4 bytes as little-endian (SPIR-V is LE)
    # file(READ ... HEX) gives bytes in order, so byte0 byte1 byte2 byte3
    # We need to reverse for uint32_t on the C side:
    # hex string: b0b1b2b3b4b5b6b7 -> bytes [b0b1] [b2b3] [b4b5] [b6b7]
    # uint32_t LE: 0xb6b7b4b5b2b3b0b1
    string(SUBSTRING "${SPV_HEX}" ${OFFSET} 2 B0)
    math(EXPR OFF1 "${OFFSET} + 2")
    string(SUBSTRING "${SPV_HEX}" ${OFF1} 2 B1)
    math(EXPR OFF2 "${OFFSET} + 4")
    string(SUBSTRING "${SPV_HEX}" ${OFF2} 2 B2)
    math(EXPR OFF3 "${OFFSET} + 6")
    string(SUBSTRING "${SPV_HEX}" ${OFF3} 2 B3)

    set(WORD "0x${B3}${B2}${B1}${B0}")

    if(I LESS LAST_WORD)
        set(WORD "${WORD},")
    endif()

    string(APPEND BODY "${WORD}")
    math(EXPR COL "${COL} + 1")
    if(COL EQUAL 8)
        string(APPEND BODY "\n    ")
        set(COL 0)
    else()
        string(APPEND BODY " ")
    endif()
endforeach()

file(WRITE "${HEADER_FILE}"
"// Auto-generated from ${SPV_FILE} — do not edit
#pragma once
#include <cstdint>
#include <vector>

static const std::vector<uint32_t> ${VAR_NAME} = {
    ${BODY}
")
file(APPEND "${HEADER_FILE}" "};\n")
