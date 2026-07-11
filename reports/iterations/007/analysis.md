# Iteration 007 analysis

Target: live arithmetic differential coverage.

The existing sibling `../moo-conformance-tests` package provides
`SocketTransport`, including Toast login, eval framing, and result parsing. The
oracle is `/root/src/toaststunt/build-release/moo` at source revision `aecc51e`,
started with Toast's own harness database `/root/src/toaststunt/test/Test.db` in
a disposable WSL directory. This iteration adds a reusable differential for arithmetic,
comparison, power, bitwise, and shift expressions before selecting any newly
exposed divergence.
