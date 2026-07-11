# Iteration 001 analysis

ToastStunt oracle: `/root/src/toaststunt` at `aecc51e`.

The full local baseline was `429 passed, 9 failed, 4 skipped`.

Target: `EOP_BITSHR` negative operands.

ToastStunt's `src/execute.cc` computes `EOP_BITSHR` as
`(UNum)lhs.v.num >> rhs.v.num`. The interpreter already implements that
unsigned 64-bit shift. The failing local test incorrectly required Python's
arithmetic right shift, so this iteration corrects the test oracle and replaces
the single example with a Hypothesis property across negative 64-bit integers
and valid nonzero shift counts.
