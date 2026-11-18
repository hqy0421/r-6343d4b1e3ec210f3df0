import copy
import re

from pymysqlreplication.util.bytes import *
from pymysqlreplication.constants import FIELD_TYPE
from enum import Enum

JSONB_TYPE_SMALL_OBJECT = 0x0
JSONB_TYPE_LARGE_OBJECT = 0x1
JSONB_TYPE_SMALL_ARRAY = 0x2
JSONB_TYPE_LARGE_ARRAY = 0x3
JSONB_TYPE_LITERAL = 0x4
JSONB_TYPE_INT16 = 0x5
JSONB_TYPE_UINT16 = 0x6
JSONB_TYPE_INT32 = 0x7
JSONB_TYPE_UINT32 = 0x8
JSONB_TYPE_INT64 = 0x9
JSONB_TYPE_UINT64 = 0xA
JSONB_TYPE_DOUBLE = 0xB
JSONB_TYPE_STRING = 0xC
JSONB_TYPE_OPAQUE = 0xF

JSONB_LITERAL_NULL = 0x0
JSONB_LITERAL_TRUE = 0x1
JSONB_LITERAL_FALSE = 0x2

JSONB_SMALL_OFFSET_SIZE = 2
JSONB_LARGE_OFFSET_SIZE = 4
JSONB_KEY_ENTRY_SIZE_SMALL = 2 + JSONB_SMALL_OFFSET_SIZE
JSONB_KEY_ENTRY_SIZE_LARGE = 2 + JSONB_LARGE_OFFSET_SIZE
JSONB_VALUE_ENTRY_SIZE_SMALL = 1 + JSONB_SMALL_OFFSET_SIZE
JSONB_VALUE_ENTRY_SIZE_LARGE = 1 + JSONB_LARGE_OFFSET_SIZE


def is_json_inline_value(type: bytes, is_small: bool) -> bool:
    if type in [JSONB_TYPE_UINT16, JSONB_TYPE_INT16, JSONB_TYPE_LITERAL]:
        return True
    elif type in [JSONB_TYPE_INT32, JSONB_TYPE_UINT32]:
        return not is_small
    return False


def parse_json(type: bytes, data: bytes):
    if type == JSONB_TYPE_SMALL_OBJECT:
        v = parse_json_object_or_array(data, True, True)
    elif type == JSONB_TYPE_LARGE_OBJECT:
        v = parse_json_object_or_array(data, False, True)
    elif type == JSONB_TYPE_SMALL_ARRAY:
        v = parse_json_object_or_array(data, True, False)
    elif type == JSONB_TYPE_LARGE_ARRAY:
        v = parse_json_object_or_array(data, False, False)
    elif type == JSONB_TYPE_LITERAL:
        v = parse_literal(data)
    elif type == JSONB_TYPE_INT16:
        v = parse_int16(data)
    elif type == JSONB_TYPE_UINT16:
        v = parse_uint16(data)
    elif type == JSONB_TYPE_INT32:
        v = parse_int32(data)
    elif type == JSONB_TYPE_UINT32:
        v = parse_uint32(data)
    elif type == JSONB_TYPE_INT64:
        v = parse_int64(data)
    elif type == JSONB_TYPE_UINT64:
        v = parse_uint64(data)
    elif type == JSONB_TYPE_DOUBLE:
        v = parse_double(data)
    elif type == JSONB_TYPE_STRING:
        length, n = decode_variable_length(data)
        v = parse_string(n, length, data)
    elif type == JSONB_TYPE_OPAQUE:
        v = parse_opaque(data)
    else:
        raise ValueError(f"Json type {type} is not handled")
    return v


def parse_json_object_or_array(bytes, is_small, is_object):
    offset_size = JSONB_SMALL_OFFSET_SIZE if is_small else JSONB_LARGE_OFFSET_SIZE
    count = decode_count(bytes, is_small)
    size = decode_count(bytes[offset_size:], is_small)
    if is_small:
        key_entry_size = JSONB_KEY_ENTRY_SIZE_SMALL
        value_entry_size = JSONB_VALUE_ENTRY_SIZE_SMALL
    else:
        key_entry_size = JSONB_KEY_ENTRY_SIZE_LARGE
        value_entry_size = JSONB_VALUE_ENTRY_SIZE_LARGE
    if is_data_short(bytes, size):
        raise ValueError(
            "Before MySQL 5.7.22, json type generated column may have invalid value"
        )

    header_size = 2 * offset_size + count * value_entry_size

    if is_object:
        header_size += count * key_entry_size

    if header_size > size:
        raise ValueError("header size > size")

    keys = []
    if is_object:
        keys = []
        for i in range(count):
            entry_offset = 2 * offset_size + key_entry_size * i
            key_offset = decode_count(bytes[entry_offset:], is_small)
            key_length = decode_uint(bytes[entry_offset + offset_size :])
            keys.append(bytes[key_offset : key_offset + key_length])

    values = {}
    for i in range(count):
        entry_offset = 2 * offset_size + value_entry_size * i
        if is_object:
            entry_offset += key_entry_size * count
        json_type = bytes[entry_offset]
        if is_json_inline_value(json_type, is_small):
            values[i] = parse_json(
                json_type, bytes[entry_offset + 1 : entry_offset + value_entry_size]
            )
            continue
        value_offset = decode_count(bytes[entry_offset + 1 :], is_small)
        if is_data_short(bytes, value_offset):
            return None
        values[i] = parse_json(json_type, bytes[value_offset:])
    if not is_object:
        return list(values.values())
    out = {}
    for i in range(count):
        out[keys[i]] = values[i]
    return out


def parse_literal(data: bytes):
    json_type = data[0]
    if json_type == JSONB_LITERAL_NULL:
        return None
    elif json_type == JSONB_LITERAL_TRUE:
        return True
    elif json_type == JSONB_LITERAL_FALSE:
        return False

    raise ValueError("NOT LITERAL TYPE")


def parse_opaque(data: bytes):
    if is_data_short(data, 1):
        return None
    type_ = data[0]
    data = data[1:]

    length, n = decode_variable_length(data)
    data = data[n : n + length]

    if type_ in [FIELD_TYPE.NEWDECIMAL, FIELD_TYPE.DECIMAL]:
        return decode_decimal(data)
    elif type_ in [FIELD_TYPE.TIME, FIELD_TYPE.TIME2]:
        return decode_time(data)
    elif type_ in [FIELD_TYPE.DATE, FIELD_TYPE.DATETIME, FIELD_TYPE.DATETIME2]:
        return decode_datetime(data)
    else:
        return data.decode(errors="ignore")


class JsonBinaryParseError(ValueError):
    """Raised when a JSON binary value or a JSON partial-update diff vector
    cannot be decoded (bad length, unknown operation, truncated payload or
    malformed path)."""


class JsonUpdateMode(Enum):
    """How partial JSON updates are exposed on row events.

    DELTA:       after_values holds a JsonDiffVector (ordered operations,
                 possibly pending materialization when no before document is
                 available). This is the default and never queries the
                 database to guess a base.
    MATERIALIZED: after_values holds the complete after JSON document whenever
                 a base is available; vectors without a base remain explicit
                 pending deltas instead of being faked as complete JSON.
    BOTH:        after_values holds the JsonDiffVector and the materialized
                 after document is additionally available on the vector and
                 under the '<column>.materialized' row key.
    """

    DELTA = "delta"
    MATERIALIZED = "materialized"
    BOTH = "both"


class JsonDiffOperation(Enum):
    # The JSON value in the given path is replaced with a new value.
    # It has the same effect as `JSON_REPLACE(col, path, value)`.
    Replace = 0
    # Add a new element at the given path.
    # If the path specifies an array element, it has the same effect as `JSON_ARRAY_INSERT(col, path, value)`.
    # If the path specifies an object member, it has the same effect as `JSON_INSERT(col, path, value)`.
    Insert = 1
    # The JSON value at the given path is removed from an array or object.
    # It has the same effect as `JSON_REMOVE(col, path)`.
    Remove = 2

    @staticmethod
    def by_index(index):
        try:
            return JsonDiffOperation(index)
        except ValueError:
            raise JsonBinaryParseError(
                f"Unknown JSON partial update operation: {index}"
            )


class JsonDiff:
    # JsonDiffOperation Remove Operation Does not have Value
    def __init__(self, op: JsonDiffOperation, path: bytes, value=None):
        self.op = op
        self.path = path
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, JsonDiff):
            return NotImplemented
        return (
            self.op == other.op
            and self.path == other.path
            and self.value == other.value
        )

    def __repr__(self):
        return f"JsonDiff(op={self.op!r}, path={self.path!r}, value={self.value!r})"

    def __str__(self):
        if isinstance(self.value, (bytes, bytearray)):
            rendered = self.value.decode()
        elif self.value is None:
            rendered = ""
        else:
            rendered = str(self.value)
        return f"JsonDiff(op :{self.op} path :{self.path.decode()} value :{rendered})"


# ---------------------------------------------------------------------------
# JSON path handling
# ---------------------------------------------------------------------------

# A MySQL JSON path is a byte string of the form
#   $           (the whole document)
#   $.member    (object member; quoted with double quotes when needed)
#   $[0]        (array index, possibly negative)
#   $[*]        (wildcard - rejected, partial updates never use it)
# followed by any number of additional legs.
_PATH_MEMBER_RE = re.compile(rb"\.(?:[A-Za-z0-9_$]+|\"(?:\\.|[^\"\\])*\")")
_PATH_INDEX_RE = re.compile(rb"\[-?[0-9]+\]")
_PATH_WILDCARD_RE = re.compile(rb"\[\*\]|\.\*")


def validate_json_path(path: bytes):
    """Validate a MySQL JSON path as serialized in a partial-update diff.

    Raises JsonBinaryParseError for empty, truncated or otherwise malformed
    paths. Wildcard paths are rejected because partial updates always target a
    single concrete location.
    """
    if not isinstance(path, (bytes, bytearray)) or len(path) == 0:
        raise JsonBinaryParseError("JSON path is empty or not a byte string")
    if path[0:1] != b"$":
        raise JsonBinaryParseError(f"JSON path must start with '$': {path!r}")
    if _PATH_WILDCARD_RE.search(path):
        raise JsonBinaryParseError(f"Wildcard JSON path is not supported: {path!r}")

    pos = 1
    length = len(path)
    while pos < length:
        member = _PATH_MEMBER_RE.match(path, pos)
        if member:
            pos = member.end()
            continue
        index = _PATH_INDEX_RE.match(path, pos)
        if index:
            pos = index.end()
            continue
        raise JsonBinaryParseError(f"Malformed JSON path: {path!r}")


class _PathLeg:
    __slots__ = ("member", "identifier")

    def __init__(self, member=None, identifier=None):
        self.member = member
        self.identifier = identifier


def parse_json_path(path: bytes):
    """Parse a validated MySQL JSON path into an ordered list of legs.

    An empty list represents the whole document ('$'). A member leg is a
    byte string key; an index leg is a (possibly negative) integer.
    """
    validate_json_path(path)
    legs = []
    pos = 1
    length = len(path)
    while pos < length:
        if path[pos : pos + 1] == b".":
            if path[pos + 1 : pos + 2] == b'"':
                end = pos + 2
                raw = bytearray()
                while end < length:
                    char = path[end]
                    if char == 0x5C:  # backslash escape
                        if end + 1 >= length:
                            raise JsonBinaryParseError(
                                f"Truncated escape in JSON path: {path!r}"
                            )
                        raw.append(path[end + 1])
                        end += 2
                    elif char == 0x22:  # closing quote
                        break
                    else:
                        raw.append(char)
                        end += 1
                if end >= length:
                    raise JsonBinaryParseError(
                        f"Unterminated quoted JSON path member: {path!r}"
                    )
                legs.append(_PathLeg(member=bytes(raw)))
                pos = end + 1
            else:
                match = _PATH_MEMBER_RE.match(path, pos)
                legs.append(_PathLeg(member=bytes(match.group(0)[1:])))
                pos = match.end()
        else:
            match = _PATH_INDEX_RE.match(path, pos)
            legs.append(_PathLeg(identifier=int(match.group(0)[1:-1])))
            pos = match.end()
    return legs


_MISSING = object()


def _resolve_parent(document, legs):
    """Walk every leg but the last and return (parent, last_leg)."""
    current = document
    for leg in legs[:-1]:
        if leg.member is not None:
            if not isinstance(current, dict) or leg.member not in current:
                return None, None
            current = current[leg.member]
        else:
            if not isinstance(current, list):
                return None, None
            idx = leg.identifier
            if idx < 0:
                idx += len(current)
            if idx < 0 or idx >= len(current):
                return None, None
            current = current[idx]
    return current, legs[-1]


def _apply_replace(document, legs, value):
    if not legs:
        return value
    parent, last = _resolve_parent(document, legs)
    if parent is None:
        return document
    if last.member is not None:
        if isinstance(parent, dict) and last.member in parent:
            parent[last.member] = value
    else:
        if isinstance(parent, list):
            idx = last.identifier
            if idx < 0:
                idx += len(parent)
            if 0 <= idx < len(parent):
                parent[idx] = value
    return document


def _apply_insert(document, legs, value):
    if not legs:
        # JSON_INSERT on the document root is a no-op.
        return document
    parent, last = _resolve_parent(document, legs)
    if parent is None:
        return document
    if last.member is not None:
        if isinstance(parent, dict) and last.member not in parent:
            parent[last.member] = value
    elif isinstance(parent, list):
        idx = last.identifier
        if idx < 0:
            idx += len(parent)
        idx = max(idx, 0)
        if idx > len(parent):
            idx = len(parent)
        parent.insert(idx, value)
    return document


def _apply_remove(document, legs):
    if not legs:
        # Removing the document root yields SQL NULL in MySQL; represent it
        # with the dedicated sentinel so it is never confused with JSON null.
        return _MISSING
    parent, last = _resolve_parent(document, legs)
    if parent is None:
        return document
    if last.member is not None:
        if isinstance(parent, dict) and last.member in parent:
            del parent[last.member]
    elif isinstance(parent, list):
        idx = last.identifier
        if idx < 0:
            idx += len(parent)
        if 0 <= idx < len(parent):
            del parent[idx]
    return document


class JsonDiffVector:
    """An ordered sequence of JsonDiff operations belonging to one JSON
    column in one row image, exactly as serialized in the binlog.

    The vector preserves order (including an empty vector) and can be
    materialized against a base document. Materialization applies the diffs
    sequentially, so every operation observes the effects of the preceding
    ones. The base document itself is never modified.
    """

    def __init__(self, diffs=None, base=None, has_base=False):
        self._diffs = list(diffs) if diffs else []
        self._base = base
        self._has_base = has_base
        self._materialized = None
        if has_base:
            self._materialize_locked()

    # -- container protocol -------------------------------------------------
    def __iter__(self):
        return iter(self._diffs)

    def __len__(self):
        return len(self._diffs)

    def __getitem__(self, index):
        return self._diffs[index]

    def __eq__(self, other):
        if not isinstance(other, JsonDiffVector):
            return NotImplemented
        return self._diffs == other._diffs

    def __repr__(self):
        return f"JsonDiffVector({self._diffs!r})"

    # -- properties ---------------------------------------------------------
    @property
    def diffs(self):
        return list(self._diffs)

    @property
    def is_empty(self):
        return len(self._diffs) == 0

    @property
    def has_base(self):
        """True when the full before document of the same row was available
        (or an explicit base was supplied) and an after document could be
        materialized."""
        return self._has_base

    @property
    def is_pending_materialization(self):
        """True when the vector holds real diffs but no base document is
        available, so a complete after document cannot be produced. The
        vector must stay a delta in that case and must never be reported as a
        complete JSON value or as SQL NULL."""
        return not self._has_base

    @property
    def materialized(self):
        """The materialized after document, or None when no base is
        available. Check has_base/is_pending_materialization first."""
        if not self._has_base:
            return None
        return self._materialized

    # -- materialization ----------------------------------------------------
    def _materialize_locked(self):
        document = copy.deepcopy(self._base)
        for diff in self._diffs:
            legs = parse_json_path(diff.path)
            if diff.op == JsonDiffOperation.Replace:
                document = _apply_replace(document, legs, diff.value)
            elif diff.op == JsonDiffOperation.Insert:
                document = _apply_insert(document, legs, diff.value)
            else:
                document = _apply_remove(document, legs)
                if document is _MISSING:
                    break
        self._materialized = None if document is _MISSING else document

    def materialize(self, base):
        """Return a new JsonDiffVector materialized against an explicit base
        document supplied by the caller. The caller's base is never mutated
        and this vector is left untouched, so a failed materialization cannot
        corrupt any existing state."""
        return JsonDiffVector(self._diffs, base=base, has_base=True)

    def __str__(self):
        return f"JsonDiffVector({[str(d) for d in self._diffs]})"


# ---------------------------------------------------------------------------
# Strict binary reader for the partial-update diff vector
# ---------------------------------------------------------------------------


class _StrictBinReader:
    """Bounds-checked cursor over the serialized diff vector payload.

    Every read validates lengths up front; any truncation, bad length or
    unknown encoding raises JsonBinaryParseError before state is handed back,
    so callers never observe a half-parsed vector."""

    def __init__(self, data: bytes):
        self._data = bytes(data)
        self._pos = 0

    def remaining(self):
        return len(self._data) - self._pos

    def read_byte(self):
        if self._pos >= len(self._data):
            raise JsonBinaryParseError("Truncated JSON diff payload (expected 1 byte)")
        value = self._data[self._pos]
        self._pos += 1
        return value

    def read(self, size):
        if size < 0:
            raise JsonBinaryParseError(f"Negative length in JSON diff payload: {size}")
        if self._pos + size > len(self._data):
            raise JsonBinaryParseError(
                "Truncated JSON diff payload: "
                f"needed {size} bytes, {self.remaining()} available"
            )
        chunk = self._data[self._pos : self._pos + size]
        self._pos += size
        return chunk

    def read_length_encoded_int(self, what):
        if self._pos >= len(self._data):
            raise JsonBinaryParseError(
                f"Truncated JSON diff payload while reading {what} length"
            )
        first = self._data[self._pos]
        if first == 251:
            raise JsonBinaryParseError(
                f"NULL marker is not valid for {what} length in JSON diff"
            )
        if first < 251:
            self._pos += 1
            return first
        if first == 252:
            size = 2
        elif first == 253:
            size = 3
        elif first == 254:
            size = 8
        else:
            raise JsonBinaryParseError(
                f"Invalid length-encoded integer prefix 0x{first:02x} for {what}"
            )
        if self._pos + 1 + size > len(self._data):
            raise JsonBinaryParseError(
                f"Truncated JSON diff payload while reading {what} length"
            )
        payload = self._data[self._pos + 1 : self._pos + 1 + size]
        value = int.from_bytes(payload, byteorder="little")
        self._pos += 1 + size
        return value

    def at_end(self):
        return self._pos == len(self._data)


def parse_json_diff_vector(data: bytes):
    """Parse the payload of a JSON column transported as a partial update.

    Wire format (MySQL 8.0 json_binary.cc serialize_json_diff_vector):
      length-encoded int : number of diffs (0 is legal - an empty vector)
      for each diff:
        1 byte operation (0=Replace, 1=Insert, 2=Remove)
        length-encoded path length + path bytes
        for Replace/Insert only:
          length-encoded value length + binary JSON value

    Raises JsonBinaryParseError on bad length, unknown operation, truncated
    payload or bad path. On success every byte is consumed, preventing a
    malformed vector from shifting the parse position of later columns.
    """
    reader = _StrictBinReader(data)
    diff_count = reader.read_length_encoded_int("diff count")
    diffs = []
    for _ in range(diff_count):
        operation = JsonDiffOperation.by_index(reader.read_byte())
        path_length = reader.read_length_encoded_int("path")
        path = reader.read(path_length)
        validate_json_path(path)
        if operation == JsonDiffOperation.Remove:
            diffs.append(JsonDiff(operation, path))
            continue
        value_length = reader.read_length_encoded_int("value")
        value_bytes = reader.read(value_length)
        if len(value_bytes) == 0:
            raise JsonBinaryParseError(
                "Truncated JSON diff payload: missing binary JSON value"
            )
        value = parse_json(value_bytes[0], value_bytes[1:])
        diffs.append(JsonDiff(operation, path, value))
    if not reader.at_end():
        raise JsonBinaryParseError(
            f"JSON diff vector has {reader.remaining()} trailing bytes"
        )
    return JsonDiffVector(diffs)
