# Iteration 007 analysis

Target: live arithmetic differential coverage.

The existing sibling `../moo-conformance-tests` package provides
`SocketTransport`, including Toast login, eval framing, and result parsing. The
oracle is `C:\Users\Q\src\toaststunt\build\moo`, rebuilt from source revision
`e8a353665a106244f5e01edb67239c90411ae584`, and started with Toast's own
harness database `C:\Users\Q\src\toaststunt\test\Test.db` in
a disposable WSL directory. This iteration adds a reusable differential for arithmetic,
comparison, power, bitwise, and shift expressions before selecting any newly
exposed divergence.

The first run used a nearby `/root/src/toaststunt` checkout. That was not the
literal requested artifact. The target source files and `Test.db` are
byte-identical across those revisions, and iteration 011 reran the completed
910-case gate against the literal checkout with zero mismatches.
