# Iteration 004 analysis

ToastStunt oracle: `C:\Users\Q\src\toaststunt` at
`e8a353665a106244f5e01edb67239c90411ae584`. The cited `src/execute.cc`
blob is byte-identical to the one originally inspected.

Target: object verb-call fixtures and any exposed call-frame defects.

ToastStunt's `src/execute.cc` lines 2115-2200 accepts any receiver, but only a
`TYPE_OBJ` value directly names an object; `TYPE_INT` is routed through
`#0.int_proto`. The local tests incorrectly described and encoded object IDs as
plain integers even though the compiler correctly emits `ObjNum` for `#N`
literals. Correcting the fixture type is necessary before remaining failures can
identify real `OP_CALL_VERB` defects.
