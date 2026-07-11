# Iteration 008 analysis

Target: live core-container differential coverage.

The arithmetic oracle gate is extended with one new family covering strings,
lists, and maps: construction, equality, membership, concatenation, indexing,
ranges, and first/last index expressions. Results are compared through the same
ToastStunt `aecc51e` process and existing `SocketTransport` parser used in
iteration 007.
