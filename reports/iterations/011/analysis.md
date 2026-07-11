# Iteration 011 analysis

Target: exact Toast oracle reconciliation and builtin namespace inventory.

Behavioral samples do not prove namespace convergence. This iteration uses the
sibling harness's existing `extract_builtin_specs()` parser on the literal
ToastStunt checkout
`C:\Users\Q\src\toaststunt@e8a353665a106244f5e01edb67239c90411ae584`
and compares those registered names with the current checkout's
`BuiltinFunctions.functions` keys.

The earlier live runner used `/root/src/toaststunt`, which was a nearby checkout
and therefore invalid provenance. The exact checkout was rebuilt and the full
910-case differential was rerun with zero mismatches. The six Toast source
blobs cited by iterations 001-005 and `test/Test.db` are byte-identical across
the two revisions; the revision delta is confined to the JSON-v20 database
backend.

The namespace inventory reports 229 Toast registrations, 223 local registered
names, 129 shared names, 100 missing names, and 94 extra names. The local
constructor currently registers every callable attribute, including private
helpers and registry-management methods. This measured namespace gap is the
authoritative next convergence surface.
