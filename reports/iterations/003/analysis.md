# Iteration 003 analysis

ToastStunt oracle: `/root/src/toaststunt` at `aecc51e`.

Target: optional scatter defaults.

ToastStunt's `src/code_gen.cc` lines 842-879 and `src/execute.cc` lines
2458-2524 show that a missing optional value jumps to ordinary compiled
bytecode for its default expression. The local VM instead special-cases a
single literal, decodes optimized integers from the wrong base, and replaces
all multi-instruction defaults with zero. The fix must execute the existing
default bytecode in the current environment.
