# Iteration 013 analysis

Target: registry infrastructure exposed as MOO builtins.

The remaining extras include `register`, five `get_*` lookup methods, and
`raise_error`. Repository search shows the lookup methods serve the compiler,
VM, debugger, and dictionary protocol; they are infrastructure owned by
`BuiltinFunctions`. `register` has no external caller. `raise_error` is only a
Python-safe implementation name for Toast's real `raise` builtin.

This slice makes the existing infrastructure methods private and registers the
raise implementation directly under `raise`. It adds no replacement interface
or compatibility alias.
