# Iteration 004 analysis

ToastStunt oracle: `/root/src/toaststunt` at `aecc51e`.

Target: object verb-call fixtures and any exposed call-frame defects.

ToastStunt's `src/execute.cc` lines 2115-2200 accepts any receiver, but only a
`TYPE_OBJ` value directly names an object; `TYPE_INT` is routed through
`#0.int_proto`. The local tests incorrectly described and encoded object IDs as
plain integers even though the compiler correctly emits `ObjNum` for `#N`
literals. Correcting the fixture type is necessary before remaining failures can
identify real `OP_CALL_VERB` defects.
