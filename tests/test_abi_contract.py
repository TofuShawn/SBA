# Copyright (c) 2026 TofuShawn
# SPDX-License-Identifier: GPL-3.0-or-later

"""ABI contract guards for the hand-written ctypes bindings.

The C/Rust structs and the exported signatures are mirrored by hand in
cengine.py / rustbridge.py. sba_flat_mcts once lost a single c_int32 in
that mirror: the whole Flat MCTS backend crashed at runtime while the
full suite stayed green, because nothing compared the two sides.

These tests rebuild the layouts from the authoritative declarations in
sba_ultt.h / lib.rs and compare field order, widths, offsets and sizeof
against the Python classes. They also check every exported function's
arity against its declaration, which is what catches a dropped argument.
No ABI changes and no rebuild are needed to run them.
"""

import ctypes
import re
from pathlib import Path

import pytest

from az import rustbridge
from engines import cengine

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src" / "engines" / "sba_ultt.h").read_text(encoding="utf-8")
RUST = (ROOT / "src" / "sba-game" / "src" / "lib.rs").read_text(encoding="utf-8")

C_TYPES = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "int32_t": ctypes.c_int32,
    "uint8_t": ctypes.c_uint8,
    "uint32_t": ctypes.c_uint32,
    "uint64_t": ctypes.c_uint64,
}

RUST_TYPES = {
    "[u16; 9]": ctypes.c_uint16 * 9,
    "u16": ctypes.c_uint16,
    "u8": ctypes.c_uint8,
    "i32": ctypes.c_int32,
    "f32": ctypes.c_float,
}


def _c_structs(text):
    """name -> field lines, for every `typedef struct { ... } Name;`."""
    return {
        m.group(2): m.group(1)
        for m in re.finditer(r"typedef struct \{(.*?)\}\s*(\w+)\s*;", text, re.S)
    }


def _c_fields(text, name):
    structs = _c_structs(text)
    assert name in structs, name
    fields = []
    for raw in structs[name].splitlines():
        line = raw.split("/*")[0].strip()
        if not line:
            continue
        m = re.match(r"(\w+)\s+(\w+)\s*(\[\d+\])?\s*;", line)
        assert m, "unparsed C field: %r" % line
        ctype, fname, array = m.groups()
        base = C_TYPES[ctype]
        fields.append((fname, base * int(array[1:-1]) if array else base))
    return fields


def _rust_fields(text, name):
    m = re.search(r"pub struct " + re.escape(name) + r"\s*\{(.*?)\n\}", text, re.S)
    assert m, name
    fields = []
    for raw in m.group(1).splitlines():
        line = raw.split("//")[0].strip()
        if not line:
            continue
        f = re.match(r"pub\s+(\w+)\s*:\s*([^,]+?)\s*,?$", line)
        assert f, "unparsed Rust field: %r" % line
        fields.append((f.group(1), RUST_TYPES[f.group(2).strip()]))
    return fields


def _assert_same_layout(parsed, handwritten, label):
    """Rebuild the struct from the declaration, then compare it to the
    hand-written ctypes class field by field."""
    auto = type("Auto" + label, (ctypes.Structure,), {"_fields_": parsed})
    assert [n for n, _ in parsed] == [n for n, _ in handwritten._fields_], label
    assert ctypes.sizeof(auto) == ctypes.sizeof(handwritten), label
    for name, _ in parsed:
        assert getattr(auto, name).offset == getattr(handwritten, name).offset, (label, name)


def test_c_search_opts_layout():
    _assert_same_layout(_c_fields(HEADER, "SbaSearchOpts"), cengine.SearchOpts, "SearchOpts")


def test_c_pro_opts_layout():
    _assert_same_layout(_c_fields(HEADER, "SbaProOpts"), cengine.ProOpts, "ProOpts")


def test_rust_state_layout():
    assert "#[repr(C)]" in RUST
    _assert_same_layout(_rust_fields(RUST, "SbaState"), rustbridge.SbaState, "SbaState")


def test_rust_abi_constant_matches_source():
    m = re.search(r"pub const ABI:\s*i32\s*=\s*(\d+)\s*;", RUST)
    assert m and int(m.group(1)) == rustbridge.ABI


def test_c_export_arities_match_the_bindings():
    lib = cengine._load()
    if lib is None or not cengine._has_engine:
        pytest.skip("full engine DLL not built")
    decls = re.findall(r"int32_t\s+__stdcall\s+(sba_\w+)\s*\((.*?)\)\s*;", HEADER, re.S)
    assert len(decls) >= 10, "header parser found too few exports"
    for name, args in decls:
        fn = getattr(lib, name)  # raises if a declared export is gone
        params = [a for a in args.split(",") if a.strip()]
        expected = 0 if params == ["void"] else len(params)
        assert fn.argtypes is not None, "%s has no argtypes" % name
        assert len(fn.argtypes) == expected, (name, expected, len(fn.argtypes))


def test_rust_export_arities_match_the_bindings():
    if not rustbridge.available():
        pytest.skip("sba_game cdylib not built")
    lib = rustbridge._LIB
    decls = re.findall(
        r'#\[no_mangle\]\s*pub extern "C" fn (\w+)\s*\((.*?)\)\s*(?:->[^{]*)?\{', RUST, re.S
    )
    assert len(decls) >= 10, "Rust parser found too few exports"
    for name, args in decls:
        fn = getattr(lib, name)
        expected = len([a for a in args.split(",") if a.strip()])
        assert fn.argtypes is not None, "%s has no argtypes" % name
        assert len(fn.argtypes) == expected, (name, expected, len(fn.argtypes))
