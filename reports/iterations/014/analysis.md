# Iteration 014 analysis

Target: non-Toast bitwise builtin functions.

`bitand`, `bitnot`, `bitor`, `bitshl`, `bitshr`, and `bitxor` are absent from
ToastStunt's registrations. Toast exposes the corresponding behavior through
language operators. Repository search finds no local callers outside these six
definitions, so exact namespace convergence requires deletion without an alias
or compatibility surface.
