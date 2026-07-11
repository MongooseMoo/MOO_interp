# Iteration Log

## 001 - 2026-07-11

- Start: 9 failures
- Target: ToastStunt unsigned right-shift semantics
- Result: 8 failures (-1)
- Commit: recorded by this commit

## 002 - 2026-07-11

- Start: 8 failures
- Target: ToastStunt `EOP_FIRST` and `EOP_LAST` boundary semantics
- Result: 6 failures (-2), with 3 additional coverage cases
- Commit: recorded by this commit

## 003 - 2026-07-11

- Start: 6 failures
- Target: ToastStunt optional scatter default evaluation
- Result: 5 failures (-1), with 2 additional coverage cases
- Commit: recorded by this commit

## 004 - 2026-07-11

- Start: 5 failures
- Target: ToastStunt object receiver typing and verb source compilation
- Result: 2 failures (-3)
- Commit: recorded by this commit

## 005 - 2026-07-11

- Start: 2 failures
- Target: ToastStunt integer exponentiation property
- Result: 1 failure (-1), with 1 additional coverage case
- Commit: recorded by this commit

## 006 - 2026-07-11

- Start: 1 failure
- Target: absent `toastcore.db` fixture handling
- Result: 0 failures (-1), 443 passed and 5 explicit skips
- Commit: recorded by this commit

## 007 - 2026-07-11

- Start: 0 local failures
- Target: live ToastStunt arithmetic differential
- Result: 810/810 oracle cases matched; local suite stayed green
- Commit: recorded by this commit

## 008 - 2026-07-11

- Start: 810 live oracle cases matched
- Target: core string/list/map differential
- Result: 845/845 oracle cases matched; local suite stayed green
- Commit: recorded by this commit

## 009 - 2026-07-11

- Start: 32 conversion/type differential mismatches
- Target: shared error and object type identity
- Result: 28 mismatches (-4), all isolated to non-Toast builtins
- Commit: recorded by this commit

## 010 - 2026-07-11

- Start: 28 mismatches from four non-Toast builtins
- Target: delete extra conversion/type builtin surface
- Result: 910/910 oracle cases matched (28 mismatches removed)
- Commit: recorded by this commit
