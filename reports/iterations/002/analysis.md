# Iteration 002 analysis

ToastStunt oracle: `C:\Users\Q\src\toaststunt` at
`e8a353665a106244f5e01edb67239c90411ae584`. The cited opcode, compiler,
and executor source blobs are byte-identical to those originally inspected.

Target: `EOP_FIRST` and `EOP_LAST`.

ToastStunt's `src/execute.cc` lines 2392-2440 show that these opcodes return
boundary indices, not container elements. Lists and strings produce `1` or `0`
for first and their length for last. Maps produce their first or last sorted key,
or `none` when empty. The opcodes `EOP_FIRST_INDEX` and `EOP_LAST_INDEX` do not
exist in ToastStunt and must be removed after their compiler callers move to the
real opcodes.
