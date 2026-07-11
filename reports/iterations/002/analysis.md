# Iteration 002 analysis

ToastStunt oracle: `/root/src/toaststunt` at `aecc51e`.

Target: `EOP_FIRST` and `EOP_LAST`.

ToastStunt's `src/execute.cc` lines 2392-2440 show that these opcodes return
boundary indices, not container elements. Lists and strings produce `1` or `0`
for first and their length for last. Maps produce their first or last sorted key,
or `none` when empty. The opcodes `EOP_FIRST_INDEX` and `EOP_LAST_INDEX` do not
exist in ToastStunt and must be removed after their compiler callers move to the
real opcodes.
