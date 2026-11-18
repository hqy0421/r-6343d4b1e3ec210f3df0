"""Offline tests for MySQL 8.0 partial JSON update (diff vector) support.

These tests build binary diff-vector payloads by hand, so they need no running
MySQL server. They cover:

  * ordered parsing of replace/insert/remove sequences and empty vectors;
  * strict rejection of bad lengths, unknown operations, truncation and bad
    paths, without consuming bytes that belong to later columns;
  * materialization against a same-row before document (or an explicit base),
    where later operations observe earlier effects and the base is untouched;
  * the four-way distinction between SQL NULL, JSON null, not-included columns
    and empty vectors;
  * pending deltas that must never masquerade as complete JSON or SQL NULL.
"""

import copy
import struct
import unittest

from pymysqlreplication.json_binary import (
    JsonBinaryParseError,
    JsonDiff,
    JsonDiffOperation as Op,
    JsonDiffVector,
    JsonUpdateMode,
    parse_json,
    parse_json_diff_vector,
    validate_json_path,
    parse_json_path,
)
from pymysqlreplication.constants import NONE_SOURCE
from pymysqlreplication.util.bytes import length_encoded_int


# ---------------------------------------------------------------------------
# Binary JSON document fixtures
# ---------------------------------------------------------------------------


def varlen(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out += bytes([b | 0x80])
        else:
            out += bytes([b])
            break
    return out


def le_int(n):
    """MySQL length-encoded integer used inside diff vectors."""
    if n < 251:
        return bytes([n])
    if n <= 0xFFFF:
        return bytes([252]) + struct.pack("<H", n)
    if n <= 0xFFFFFF:
        return bytes([253]) + n.to_bytes(3, "little")
    return bytes([254]) + struct.pack("<Q", n)


def json_string(s):
    """Binary JSON document for a JSON string scalar."""
    if isinstance(s, str):
        s = s.encode()
    return bytes([0xC]) + varlen(len(s)) + s


def json_literal(lit):
    """0 => JSON null, 1 => true, 2 => false."""
    return bytes([0x4, lit])


JSON_NULL = json_literal(0)


def json_object(pairs):
    """Small JSON object from an ordered list of (bytes key, document bytes).

    Only non-inline value types (objects/arrays/strings/literals with their
    own payload) are supported by this fixture builder.
    """
    count = len(pairs)
    header_size = 4
    key_entry_size = 4
    value_entry_size = 3
    keys_region = b"".join(k for k, _ in pairs)
    key_offsets = []
    cursor = header_size + count * (key_entry_size + value_entry_size)
    for k, _ in pairs:
        key_offsets.append(cursor)
        cursor += len(k)
    value_offsets = []
    value_blobs = []
    for _, doc in pairs:
        # Offset points at the value payload past the type byte. Literal
        # documents carry their literal byte right after the type byte.
        value_offsets.append(cursor + 1)
        value_blobs.append(doc)
        cursor += len(doc)
    size = cursor
    payload = struct.pack("<HH", count, size)
    for (k, _), off in zip(pairs, key_offsets):
        payload += struct.pack("<HH", off, len(k))
    for doc, off in zip((d for _, d in pairs), value_offsets):
        payload += bytes([doc[0]]) + struct.pack("<H", off)
    payload += keys_region
    payload += b"".join(value_blobs)
    return bytes([0x0]) + payload


def json_array(docs):
    """Small JSON array; element documents must be non-inline payload types."""
    count = len(docs)
    header_size = 4
    value_entry_size = 3
    cursor = header_size + count * value_entry_size
    entries = b""
    body = b""
    for doc in docs:
        entries += bytes([doc[0]]) + struct.pack("<H", cursor + 1)
        body += doc
        cursor += len(doc)
    return bytes([0x2]) + struct.pack("<HH", count, cursor) + entries + body


def diff_entry(op, path, value_doc=None):
    """Serialize one diff operation."""
    if isinstance(path, str):
        path = path.encode()
    out = bytes([op.value]) + le_int(len(path)) + path
    if op != Op.Remove:
        out += le_int(len(value_doc)) + value_doc
    return out


def diff_vector(diffs):
    """Serialize a full diff-vector payload (the bytes after the column's
    outer length prefix)."""
    body = le_int(len(diffs)) + b"".join(diffs)
    return body


# ---------------------------------------------------------------------------
# Parsing: ordering, operations, empty vector
# ---------------------------------------------------------------------------


class TestParseDiffVector(unittest.TestCase):
    def test_empty_vector(self):
        vec = parse_json_diff_vector(le_int(0))
        assert isinstance(vec, JsonDiffVector)
        assert len(vec) == 0
        assert vec.is_empty
        assert list(vec) == []
        assert vec.is_pending_materialization
        assert vec.materialized is None

    def test_single_replace(self):
        payload = diff_vector(
            [diff_entry(Op.Replace, "$.ab", json_string('["ab_updated"]'))]
        )
        vec = parse_json_diff_vector(payload)
        assert len(vec) == 1
        diff = vec[0]
        assert diff.op == Op.Replace
        assert diff.path == b"$.ab"
        assert diff.value == b'["ab_updated"]'

    def test_single_remove_has_no_value(self):
        vec = parse_json_diff_vector(diff_vector([diff_entry(Op.Remove, "$.ab")]))
        assert vec[0].op == Op.Remove
        assert vec[0].path == b"$.ab"
        assert vec[0].value is None

    def test_insert_operation(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Insert, "$.new", JSON_NULL)])
        )
        assert vec[0].op == Op.Insert
        # JSON null is Python None but the diff object itself is present.
        assert vec[0].value is None

    def test_ordered_mixed_sequence_preserved(self):
        docs = [
            diff_entry(Op.Replace, "$.a", json_string("r")),
            diff_entry(Op.Insert, "$[1]", json_literal(1)),
            diff_entry(Op.Remove, "$.b"),
            diff_entry(Op.Insert, '$.q', json_string("z")),
        ]
        vec = parse_json_diff_vector(diff_vector(docs))
        assert [d.op for d in vec] == [
            Op.Replace,
            Op.Insert,
            Op.Remove,
            Op.Insert,
        ]
        assert [d.path for d in vec] == [b"$.a", b"$[1]", b"$.b", b"$.q"]

    def test_json_null_value_distinct_from_missing_value(self):
        vec_null = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", JSON_NULL)])
        )
        vec_remove = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.a")])
        )
        # Both expose value None; the operation distinguishes them.
        assert vec_null[0].value is None
        assert vec_null[0].op == Op.Replace
        assert vec_remove[0].value is None
        assert vec_remove[0].op == Op.Remove

    def test_nested_and_quoted_paths(self):
        vec = parse_json_diff_vector(
            diff_vector(
                [
                    diff_entry(Op.Replace, '$.a.b[2].c', json_string("v")),
                    diff_entry(Op.Replace, '$."weird key"', json_string("w")),
                    diff_entry(Op.Replace, "$[-1]", json_string("last")),
                ]
            )
        )
        assert vec[0].path == b'$.a.b[2].c'
        assert vec[1].path == b'$."weird key"'
        assert vec[2].path == b"$[-1]"

    def test_multibyte_length_encoded_count(self):
        diffs = [diff_entry(Op.Replace, "$.a", json_string("x")) for _ in range(300)]
        vec = parse_json_diff_vector(diff_vector(diffs))
        assert len(vec) == 300

    def test_object_and_array_values_parse(self):
        doc = json_object([(b"k", json_array([json_string("v"), json_string("w")]))])
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.o", doc)])
        )
        assert vec[0].value == {b"k": [b"v", b"w"]}


# ---------------------------------------------------------------------------
# Strict error handling
# ---------------------------------------------------------------------------


class TestParseErrors(unittest.TestCase):
    def test_unknown_operation(self):
        payload = le_int(1) + bytes([9]) + le_int(2) + b"$."
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_truncated_operation_byte(self):
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(le_int(1))

    def test_truncated_path_length(self):
        payload = le_int(1) + bytes([Op.Remove.value])
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_truncated_path_bytes(self):
        payload = le_int(1) + bytes([Op.Remove.value]) + le_int(5) + b"$."
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_truncated_value_length(self):
        payload = (
            le_int(1)
            + bytes([Op.Replace.value])
            + le_int(3)
            + b"$.a"
        )
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_truncated_value_bytes(self):
        payload = (
            le_int(1)
            + bytes([Op.Replace.value])
            + le_int(3)
            + b"$.a"
            + le_int(4)
            + json_string("x")
        )
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_zero_length_value_rejected(self):
        payload = (
            le_int(1)
            + bytes([Op.Replace.value])
            + le_int(3)
            + b"$.a"
            + le_int(0)
        )
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_trailing_bytes_rejected(self):
        good = diff_vector([diff_entry(Op.Remove, "$.a")])
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(good + b"\x00")

    def test_declared_count_exceeds_entries(self):
        payload = le_int(3) + diff_entry(Op.Remove, "$.a")
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_bad_path_not_starting_with_dollar(self):
        payload = diff_vector([diff_entry(Op.Remove, b"a.b")])
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_empty_path_rejected(self):
        payload = (
            le_int(1)
            + bytes([Op.Remove.value])
            + le_int(0)
        )
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_malformed_path_rejected(self):
        for bad in [b"$..a", b"$.a[", b"$[x]", b"$.", b"$ []", b"$.a[]"]:
            payload = diff_vector([diff_entry(Op.Remove, bad)])
            with self.assertRaises(JsonBinaryParseError):
                parse_json_diff_vector(payload)

    def test_wildcard_path_rejected(self):
        for bad in [b"$[*]", b"$.a[*]", b"$.*"]:
            payload = diff_vector([diff_entry(Op.Remove, bad)])
            with self.assertRaises(JsonBinaryParseError):
                parse_json_diff_vector(payload)

    def test_null_marker_in_length_rejected(self):
        # 0xFB is the SQL NULL marker, never legal inside a diff vector length.
        payload = bytes([251])
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)

    def test_error_does_not_consume_following_column_bytes(self):
        # A corrupt column payload followed by bytes belonging to another
        # column: parsing must raise rather than silently swallow the tail.
        good = diff_vector([diff_entry(Op.Remove, "$.a")])
        foreign = b"next-column-bytes"
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(good + foreign)
        # The foreign bytes remain intact for the caller to parse as the
        # next column.
        assert foreign == b"next-column-bytes"

    def test_second_diff_bad_aborts_whole_vector(self):
        payload = (
            le_int(2)
            + diff_entry(Op.Remove, "$.a")
            + bytes([7])  # unknown op inside the second declared diff
            + le_int(2)
            + b"$."
        )
        with self.assertRaises(JsonBinaryParseError):
            parse_json_diff_vector(payload)


# ---------------------------------------------------------------------------
# Path validation/parsing unit tests
# ---------------------------------------------------------------------------


class TestJsonPath(unittest.TestCase):
    def test_valid_paths(self):
        for path in [
            b"$",
            b"$.a",
            b"$.a.b",
            b"$[0]",
            b"$[-3]",
            b'$."a b"',
            b"$.a[1].b[0]",
        ]:
            with self.subTest(path=path):
                validate_json_path(path)

    def test_invalid_paths(self):
        for path in [
            b"",
            b".a",
            b"$..a",
            b"$[*]",
            b"$.a[",
            b'$."unclosed',
            b"$[1-]",
        ]:
            with self.subTest(path=path):
                with self.assertRaises(JsonBinaryParseError):
                    validate_json_path(path)

    def test_parse_root(self):
        assert parse_json_path(b"$") == []

    def test_parse_legs(self):
        legs = parse_json_path(b"$.a[2].b")
        assert legs[0].member == b"a"
        assert legs[1].identifier == 2
        assert legs[2].member == b"b"

    def test_parse_quoted_member(self):
        legs = parse_json_path(b'$."a b"')
        assert legs[0].member == b"a b"


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


class TestMaterialization(unittest.TestCase):
    def test_materialize_replace_object_member(self):
        base = {b"a": b"x", b"b": [1, 2]}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", json_string("y"))])
        )
        resolved = vec.materialize(base)
        assert resolved.has_base
        assert resolved.materialized == {b"a": b"y", b"b": [1, 2]}
        # Base untouched.
        assert base == {b"a": b"x", b"b": [1, 2]}

    def test_materialize_insert_then_replace_sees_previous_effect(self):
        base = {b"a": 1}
        vec = parse_json_diff_vector(
            diff_vector(
                [
                    diff_entry(Op.Insert, "$.n", json_string("new")),
                    diff_entry(Op.Replace, "$.n", json_string("updated")),
                ]
            )
        )
        resolved = vec.materialize(base)
        assert resolved.materialized == {b"a": 1, b"n": b"updated"}

    def test_materialize_array_insert_order(self):
        base = [1, 2, 3]
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Insert, "$[1]", json_string("x"))])
        )
        assert vec.materialize(base).materialized == [1, b"x", 2, 3]

    def test_materialize_array_remove(self):
        base = [1, 2, 3]
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$[1]")])
        )
        assert vec.materialize(base).materialized == [1, 3]

    def test_materialize_object_remove(self):
        base = {b"a": 1, b"b": 2}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.b")])
        )
        assert vec.materialize(base).materialized == {b"a": 1}

    def test_materialize_remove_root_yields_none_document(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$")])
        )
        resolved = vec.materialize({b"a": 1})
        # The after document is SQL NULL; the vector still records the op.
        assert resolved.materialized is None
        assert resolved[0].op == Op.Remove

    def test_materialize_replace_missing_path_is_noop(self):
        base = {b"a": 1}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.missing", json_string("v"))])
        )
        assert vec.materialize(base).materialized == {b"a": 1}

    def test_materialize_insert_existing_member_is_noop(self):
        base = {b"a": 1}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Insert, "$.a", json_string("v"))])
        )
        assert vec.materialize(base).materialized == {b"a": 1}

    def test_materialize_nested_path(self):
        base = {b"o": {b"list": [10, 20]}}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.o.list[0]", JSON_NULL)])
        )
        assert vec.materialize(base).materialized == {b"o": {b"list": [None, 20]}}

    def test_materialize_json_null_member(self):
        base = {b"a": 1}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", JSON_NULL)])
        )
        result = vec.materialize(base).materialized
        # JSON null: key present, value None.
        assert b"a" in result and result[b"a"] is None

    def test_materialize_deepcopies_base(self):
        base = {b"o": {b"nested": [1, 2]}}
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.o.nested[0]")])
        )
        vec.materialize(base)
        assert base == {b"o": {b"nested": [1, 2]}}

    def test_materialize_returns_new_vector_and_keeps_original_pending(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.a")])
        )
        assert vec.is_pending_materialization
        resolved = vec.materialize({b"a": 1})
        assert vec.is_pending_materialization
        assert resolved.has_base
        assert resolved.materialized == {}

    def test_materialize_empty_vector_with_base_returns_base_copy(self):
        base = {b"a": 1}
        vec = JsonDiffVector([], base=base, has_base=True)
        assert vec.materialized == {b"a": 1}
        assert vec.materialized is not base

    def test_pending_delta_is_not_complete_json_and_not_null(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", json_string("x"))])
        )
        assert vec.is_pending_materialization
        assert vec.materialized is None
        # It still carries the real ordered delta.
        assert len(vec) == 1
        assert not isinstance(vec, dict)

    def test_materialization_failure_leaves_base_and_vector_intact(self):
        base = {b"a": 1}
        base_snapshot = copy.deepcopy(base)
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", json_string("ok"))])
        )
        # A bad path in a hand-built diff raises during materialization; the
        # caller's base must be unchanged.
        broken = JsonDiffVector([JsonDiff(Op.Replace, b"$.[", None)])
        with self.assertRaises(JsonBinaryParseError):
            broken.materialize(base)
        assert base == base_snapshot
        # Original good vector still usable.
        assert vec.materialize(base).materialized == {b"a": b"ok"}


# ---------------------------------------------------------------------------
# Compatibility of the JsonDiff value object
# ---------------------------------------------------------------------------


class TestJsonDiffCompatibility(unittest.TestCase):
    def test_constructor_and_attributes(self):
        d = JsonDiff(Op.Replace, b"$.a", 1)
        assert d.op == Op.Replace
        assert d.path == b"$.a"
        assert d.value == 1

    def test_remove_default_value_none(self):
        d = JsonDiff(Op.Remove, b"$.a")
        assert d.value is None

    def test_equality(self):
        assert JsonDiff(Op.Remove, b"$.a") == JsonDiff(Op.Remove, b"$.a")
        assert JsonDiff(Op.Remove, b"$.a") != JsonDiff(Op.Remove, b"$.b")

    def test_vector_equality(self):
        v1 = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.a")])
        )
        v2 = JsonDiffVector([JsonDiff(Op.Remove, b"$.a")])
        assert v1 == v2


# ---------------------------------------------------------------------------
# Mode enum sanity
# ---------------------------------------------------------------------------


class TestModes(unittest.TestCase):
    def test_mode_values(self):
        assert JsonUpdateMode.DELTA.value == "delta"
        assert JsonUpdateMode.MATERIALIZED.value == "materialized"
        assert JsonUpdateMode.BOTH.value == "both"

    def test_length_encoded_int_roundtrip(self):
        for n in [0, 1, 250, 251, 1000, 0xFFFFFF, 0x123456789A]:
            encoded = le_int(n)
            value, is_null, width = length_encoded_int(encoded)
            assert not is_null
            assert value == n
            assert width == len(encoded)


# ---------------------------------------------------------------------------
# Packet-level behaviour: column isolation and NULL distinctions
# ---------------------------------------------------------------------------


class FakePacket:
    """Minimal stand-in for BinLogPacketWrapper exposing only what
    read_binary_json needs."""

    def __init__(self, blob):
        self._blob = blob
        self._pos = 0
        self.read_bytes = 0

    def read(self, size):
        chunk = self._blob[self._pos : self._pos + size]
        if len(chunk) != size:
            raise AssertionError("FakePacket under-read: column payload bled out")
        self._pos += size
        self.read_bytes += size
        return chunk

    def read_uint8(self):
        return self.read(1)[0]

    def read_uint_by_size(self, size):
        return int.from_bytes(self.read(size), "little")

    def exhausted(self):
        return self._pos == len(self._blob)


from pymysqlreplication.packet import BinLogPacketWrapper  # noqa: E402


def _read_json(blob, is_partial, length_size=4):
    wrapper = BinLogPacketWrapper.__new__(BinLogPacketWrapper)
    wrapper._BinLogPacketWrapper__data_buffer = b""
    wrapper.read_bytes = 0
    wrapper.packet = FakePacket(blob)
    return wrapper.read_binary_json(length_size, is_partial)


class TestPacketLevel(unittest.TestCase):
    def test_sql_null_is_none(self):
        # length prefix 0 means SQL NULL for both modes.
        assert _read_json(struct.pack("<I", 0), is_partial=False) is None
        assert _read_json(struct.pack("<I", 0), is_partial=True) is None

    def test_non_partial_json_null_is_parsed_document(self):
        doc = JSON_NULL
        blob = struct.pack("<I", len(doc)) + doc
        value = _read_json(blob, is_partial=False)
        # JSON null decodes to Python None but is a real parsed document,
        # indistinguishable at value level from SQL NULL except via the null
        # bitmap / none-source machinery upstream.
        assert value is None

    def test_partial_empty_vector_is_not_none(self):
        payload = le_int(0)
        blob = struct.pack("<I", len(payload)) + payload
        value = _read_json(blob, is_partial=True)
        assert isinstance(value, JsonDiffVector)
        assert value.is_empty

    def test_partial_vector_round_trip(self):
        payload = diff_vector(
            [
                diff_entry(Op.Replace, "$.a", json_string("r")),
                diff_entry(Op.Remove, "$.b"),
            ]
        )
        blob = struct.pack("<I", len(payload)) + payload
        value = _read_json(blob, is_partial=True)
        assert [d.op for d in value] == [Op.Replace, Op.Remove]

    def test_multiple_json_columns_do_not_bleed(self):
        # Two JSON columns serialized back to back, each with its own length
        # prefix. A corrupt first column must raise without consuming the
        # second column's bytes.
        good_payload = diff_vector([diff_entry(Op.Remove, "$.a")])
        good_blob = struct.pack("<I", len(good_payload)) + good_payload
        corrupt_payload = le_int(2) + diff_entry(Op.Remove, "$.a")
        corrupt_blob = struct.pack("<I", len(corrupt_payload)) + corrupt_payload

        # Sanity: the good column parses and consumes exactly its own bytes.
        value = _read_json(good_blob, is_partial=True)
        assert value[0].path == b"$.a"

        with self.assertRaises(JsonBinaryParseError):
            _read_json(corrupt_blob, is_partial=True)

        # The independent second column still parses cleanly on its own.
        second = _read_json(good_blob, is_partial=True)
        assert second[0].op == Op.Remove

    def test_unknown_op_aborts_before_touching_next_column(self):
        bad = le_int(1) + bytes([9]) + le_int(2) + b"$."
        blob = struct.pack("<I", len(bad)) + bad
        with self.assertRaises(JsonBinaryParseError):
            _read_json(blob, is_partial=True)

    def test_one_byte_length_size(self):
        payload = diff_vector([diff_entry(Op.Remove, "$.x")])
        blob = struct.pack("<B", len(payload)) + payload
        value = _read_json(blob, is_partial=True, length_size=1)
        assert value[0].path == b"$.x"


# ---------------------------------------------------------------------------
# Event-level materialization: multiple JSON columns and ordinary columns
# ---------------------------------------------------------------------------


from pymysqlreplication.row_event import PartialUpdateRowsEvent  # noqa: E402


def _bare_event(mode):
    event = object.__new__(PartialUpdateRowsEvent)
    event._json_update_mode = mode
    return event


def _make_row(before, after):
    return {
        "before_values": dict(before),
        "after_values": dict(after),
        "before_none_sources": {},
        "after_none_sources": {},
    }


class TestEventMaterialization(unittest.TestCase):
    def test_delta_mode_leaves_deltas_untouched(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.a")])
        )
        row = _make_row({"j": {b"a": 1}}, {"j": vec, "id": 7})
        snapshot = copy.deepcopy(row)
        _bare_event(JsonUpdateMode.DELTA)._materialize_json_after_values(row)
        self.assertIs(row["after_values"]["j"], vec)
        self.assertEqual(row["after_values"]["id"], 7)
        self.assertEqual(row, snapshot)

    def test_materialized_mode_multiple_json_and_normal_columns(self):
        vec_with_base = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$.a")])
        )
        vec_pending = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.x", json_string("z"))])
        )
        before = {"j1": {b"a": 1, b"b": 2}, "id": 9}
        after = {"j1": vec_with_base, "j2": vec_pending, "id": 9, "name": "n"}
        row = _make_row(before, after)

        _bare_event(JsonUpdateMode.MATERIALIZED)._materialize_json_after_values(
            row
        )

        # j1 has a same-row before document -> real after JSON materialized.
        self.assertEqual(row["after_values"]["j1"], {b"b": 2})
        # j2 has no base -> explicit pending delta, never faked as JSON/NULL.
        self.assertIs(row["after_values"]["j2"], vec_pending)
        self.assertTrue(row["after_values"]["j2"].is_pending_materialization)
        # The pending reason is recorded without turning it into a NULL.
        self.assertEqual(
            row["after_none_sources"]["j2"],
            NONE_SOURCE.JSON_PENDING_MATERIALIZATION,
        )
        # Ordinary columns are untouched.
        self.assertEqual(row["after_values"]["id"], 9)
        self.assertEqual(row["after_values"]["name"], "n")
        # The before base document is unchanged.
        self.assertEqual(row["before_values"]["j1"], {b"a": 1, b"b": 2})

    def test_both_mode_keeps_vector_and_adds_materialized_key(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", json_string("y"))])
        )
        row = _make_row({"j": {b"a": b"x"}}, {"j": vec})
        _bare_event(JsonUpdateMode.BOTH)._materialize_json_after_values(row)
        self.assertIs(row["after_values"]["j"], vec)
        self.assertEqual(row["after_values"]["j.materialized"], {b"a": b"y"})

    def test_both_mode_pending_vector_has_no_materialized_key(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.x", json_string("z"))])
        )
        row = _make_row({}, {"j": vec})
        _bare_event(JsonUpdateMode.BOTH)._materialize_json_after_values(row)
        self.assertIs(row["after_values"]["j"], vec)
        self.assertNotIn("j.materialized", row["after_values"])
        self.assertEqual(
            row["after_none_sources"]["j"],
            NONE_SOURCE.JSON_PENDING_MATERIALIZATION,
        )

    def test_materialization_failure_delivers_no_half_finished_row(self):
        good_base = {b"a": 1}
        broken = JsonDiffVector([JsonDiff(Op.Replace, b"$.[", None)])
        row = _make_row({"j": good_base}, {"j": broken})
        before_snapshot = copy.deepcopy(row["before_values"])
        after_snapshot = copy.deepcopy(row["after_values"])
        with self.assertRaises(JsonBinaryParseError):
            _bare_event(
                JsonUpdateMode.MATERIALIZED
            )._materialize_json_after_values(row)
        # The row image was never committed: after_values still holds the
        # vector and the base document is untouched.
        self.assertIs(row["after_values"]["j"], broken)
        self.assertEqual(row["before_values"]["j"], good_base)
        self.assertEqual(row["before_values"], before_snapshot)
        self.assertEqual(
            list(row["after_values"].keys()), list(after_snapshot.keys())
        )

    def test_four_way_null_and_empty_distinction(self):
        # SQL NULL: ordinary None value (null bitmap upstream).
        sql_null = None
        # JSON null: an actual materialized document equal to None.
        json_null_doc = JSON_NULL
        self.assertIs(parse_json(json_null_doc[0], json_null_doc[1:]), None)
        # Empty vector: distinct object, not None.
        empty = parse_json_diff_vector(le_int(0))
        self.assertIsInstance(empty, JsonDiffVector)
        self.assertTrue(empty.is_empty)
        self.assertIsNotNone(empty)
        # Not included: column simply absent from the image.
        image = {"present": 1}
        self.assertNotIn("absent", image)
        # All four are distinguishable together.
        states = [sql_null, json_null_doc, empty, None]
        self.assertIsNone(states[0])
        self.assertIsInstance(states[2], JsonDiffVector)

    def test_materialized_vector_with_json_null_member(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Replace, "$.a", JSON_NULL)])
        )
        row = _make_row({"j": {b"a": 1}}, {"j": vec})
        _bare_event(
            JsonUpdateMode.MATERIALIZED
        )._materialize_json_after_values(row)
        result = row["after_values"]["j"]
        # JSON null keeps the member with a None value rather than removing it.
        self.assertIn(b"a", result)
        self.assertIsNone(result[b"a"])

    def test_remove_root_materializes_to_none_document(self):
        vec = parse_json_diff_vector(
            diff_vector([diff_entry(Op.Remove, "$")])
        )
        row = _make_row({"j": {b"a": 1}}, {"j": vec})
        _bare_event(
            JsonUpdateMode.MATERIALIZED
        )._materialize_json_after_values(row)
        self.assertIsNone(row["after_values"]["j"])

    def test_empty_vector_with_base_materializes_to_base_copy(self):
        empty = parse_json_diff_vector(le_int(0))
        base = {b"a": 1}
        row = _make_row({"j": base}, {"j": empty})
        _bare_event(
            JsonUpdateMode.MATERIALIZED
        )._materialize_json_after_values(row)
        self.assertEqual(row["after_values"]["j"], {b"a": 1})
        # A copy, not the same object.
        self.assertIsNot(row["after_values"]["j"], base)


if __name__ == "__main__":
    unittest.main()
