# Iteration 012 analysis

Target: private helper leakage into the builtin namespace.

`BuiltinFunctions.__init__()` registers every callable attribute whose name
does not begin with two underscores. That exposes 23 implementation helpers
whose names begin with one underscore as MOO builtins. ToastStunt registers
builtins explicitly and contains none of those names.

This slice changes only the existing registration predicate so private
callables remain ordinary implementation methods. Public non-Toast names and
missing Toast builtins are separate later families.
