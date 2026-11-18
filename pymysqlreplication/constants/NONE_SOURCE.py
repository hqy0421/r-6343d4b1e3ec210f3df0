NULL = "null"
OUT_OF_DATE_RANGE = "out of date range"
OUT_OF_DATETIME_RANGE = "out of datetime range"
OUT_OF_DATETIME2_RANGE = "out of datetime2 range"
EMPTY_SET = "empty set"
COLS_BITMAP = "cols bitmap"
JSON_PARTIAL_UPDATE = "same with before values"
# A partial JSON update decoded as an ordered diff vector that cannot be
# materialized into a complete after document because no base (the same row's
# complete before document or a caller-supplied base) is available. The value
# is an explicit pending delta - never a complete JSON document or SQL NULL.
JSON_PENDING_MATERIALIZATION = "pending json materialization"
