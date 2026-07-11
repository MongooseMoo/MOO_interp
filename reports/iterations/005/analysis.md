# Iteration 005 analysis

ToastStunt oracle: `C:\Users\Q\src\toaststunt` at
`e8a353665a106244f5e01edb67239c90411ae584`. The cited `src/numbers.cc`
blob is byte-identical to the one originally inspected.

Target: integer exponentiation property.

ToastStunt's `src/numbers.cc` lines 347-382 defines integer negative powers as
`E_DIV` for base zero, one for base one, zero for other magnitudes, and the
source's explicit parity result for base negative one. The local implementation
already matches. The failing Hypothesis test incorrectly used Python `pow` as
its oracle and did not catch the expected MOO exception from zero to a negative
power.
