# Iteration 010 analysis

Target: delete conversion/type builtins absent from ToastStunt.

`typename`, `is_type`, `tonum`, and `toerr` have no callers outside their own
definitions and the live differential. The literal ToastStunt checkout at
`C:\Users\Q\src\toaststunt@e8a353665a106244f5e01edb67239c90411ae584` has no
registrations for them. They must be deleted, not hidden. Toast also rejects an
unknown builtin while compiling a verb, so the compiler must reject a missing
builtin ID instead of emitting `OP_BI_FUNC_CALL` with operand `None`.
