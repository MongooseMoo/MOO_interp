# Iteration 010 analysis

Target: delete conversion/type builtins absent from ToastStunt.

`typename`, `is_type`, `tonum`, and `toerr` have no callers outside their own
definitions and the live differential. ToastStunt `aecc51e` has no
registrations for them. They must be deleted, not hidden. Toast also rejects an
unknown builtin while compiling a verb, so the compiler must reject a missing
builtin ID instead of emitting `OP_BI_FUNC_CALL` with operand `None`.
