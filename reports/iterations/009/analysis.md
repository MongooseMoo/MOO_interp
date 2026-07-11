# Iteration 009 analysis

Target: live conversion and type-inspection builtin differential.

The sibling Toast builtin ledger reports 7,601/7,601 required call shapes
covered. This iteration selects one pure overlapping family for direct current
interpreter comparison: `typeof`, `typename`, `is_type`, `toint`, `tofloat`,
`tonum`, `toobj`, `toerr`, `tostr`, and `toliteral`, including valid and invalid
inputs across core MOO types.

The completed differential was revalidated in iteration 011 against the
literal ToastStunt checkout at
`C:\Users\Q\src\toaststunt@e8a353665a106244f5e01edb67239c90411ae584`.
