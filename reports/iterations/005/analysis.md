# Iteration 005 analysis

ToastStunt oracle: `/root/src/toaststunt` at `aecc51e`.

Target: integer exponentiation property.

ToastStunt's `src/numbers.cc` lines 347-382 defines integer negative powers as
`E_DIV` for base zero, one for base one, zero for other magnitudes, and the
source's explicit parity result for base negative one. The local implementation
already matches. The failing Hypothesis test incorrectly used Python `pow` as
its oracle and did not catch the expected MOO exception from zero to a negative
power.
