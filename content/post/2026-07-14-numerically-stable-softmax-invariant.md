---
title: "Why Softmax Can Break—and How to Make It Safe"
date: 2026-07-14T07:00:00-04:00
description: "A plain-language guide to turning model scores into probabilities without numbers becoming too large or invalid."
slug: "numerically-stable-softmax-invariant"
tags:
  - softmax
  - numerical-stability
  - floating-point
  - pytorch
  - transformers
  - diagnostics
categories:
  - Numerical computation
series:
  - Foundation Model Engineering
curriculum_stage: "Foundations — Numerical Computation"
prerequisites:
  - "Basic arithmetic; the small amount of algebra is explained where used"
  - "Comfort reading short Python examples is helpful but not required"
  - "No prior machine-learning or numerical-computing knowledge required"
reinforces:
  - "Distributed foundation-model training"
  - "Transformer attention"
  - "Model evaluation and debugging"
introduces:
  - "Logits and softmax probabilities"
  - "Numerical invariants"
  - "Stable softmax and log-softmax"
  - "Rules for rows with no valid choices"
builds_toward:
  - "Mixed-precision error analysis"
  - "Stable attention kernels"
  - "Distributed loss computation"
params:
  math: true
draft: true
---

![The direct calculation creates a number that is too large, while subtracting the largest score first produces valid probabilities](/images/softmax-low-precision-summary-v2.png)

Suppose an image model must decide whether a picture shows a **cat**, **dog**, or **rabbit**. It starts with three scores:

```text
cat:     2.0
dog:     1.0
rabbit:  0.0
```

These raw scores are called **logits**. A larger score means the model prefers that choice. But a logit is not a probability. It can be negative, greater than one, and the scores do not have to add up to anything useful.

**Softmax** turns these scores into probabilities. It uses an operation called the **exponential**, written as \(\exp(x)=e^x\). The exponential makes every value positive and makes larger scores grow much faster than smaller ones. For example, `exp(2)` is about 7.39, while `exp(12)` is about 162,755.

Softmax calculates the exponential of every score and then divides each result by the total:

\[
p_i = \frac{\exp(x_i)}{\sum_{j=1}^{n}\exp(x_j)}.
\]

Here, \(x_i\) means one score, and \(p_i\) means the probability produced for that score. For `[2, 1, 0]`, the calculation gives:

| Label | Logit | Exponential | Probability |
|---|---:|---:|---:|
| Cat | 2 | 7.39 | 66.5% |
| Dog | 1 | 2.72 | 24.5% |
| Rabbit | 0 | 1.00 | 9.0% |

The probabilities add up to 100%. Cat gets the highest probability because it had the highest score.

This looks simple. Yet a direct translation of the formula into code can fail on a normal input.

## A small input that breaks the obvious implementation

Now use `[12, 11, 10]`. Each score is exactly 10 larger than before, but the gaps between the scores are unchanged. The first score is still one above the second and two above the third. Softmax should therefore give the same answer: about `[0.665, 0.245, 0.090]`.

The problem appears when the computer uses **16-bit floating point**, usually shortened to **FP16**. FP16 uses 16 bits in total: one stores the sign, five store the exponent that controls the size range, and ten store the fraction that controls detail. FP32 also names the total storage size, not the number of exact digits. FP16 uses less memory than FP32 and can make model training and prediction faster. But it cannot hold a finite value larger than 65,504.

The first exponential is larger than that limit:

\[
\exp(12) \approx 162{,}755.
\]

The value does not fit. This is called **overflow**. The computer stores positive infinity instead:

\[
[\exp(12), \exp(11), \exp(10)]
\rightarrow
[\infty, 59872, 22032].
\]

Softmax must divide each value by their total. The first division becomes infinity divided by infinity. That has no valid numerical answer, so the computer returns **NaN**, short for “not a number.”

| Implementation | Intermediate values | Output |
|---|---|---|
| Direct FP16 softmax | Contains infinity | `[NaN, 0, 0]` |
| Stable softmax | All exponentials are at most 1 | `[0.665, 0.245, 0.090]` |

![Direct FP16 softmax creates a number that is too large, while max-shifted FP32 softmax remains valid](/images/stable-softmax-numerical-contract.svg)

Nothing was wrong with the model's intended answer. A temporary value created during the calculation was simply too large for FP16.

## Fix the calculation before it overflows

The fix is to subtract the largest score from every score before calculating any exponentials:

```text
[12, 11, 10] - 12 = [0, -1, -2]
```

The new scores still have the same gaps: one between the first two and two between the first and third. But their exponentials are now small enough to store safely:

\[
[\exp(0), \exp(-1), \exp(-2)]
=
[1, 0.368, 0.135].
\]

After dividing each value by the total, the answer is still `[0.665, 0.245, 0.090]`.

The next equation shows why this works. Let \(c\) be the number subtracted from every score:

\[
\frac{\exp(x_i-c)}{\sum_j \exp(x_j-c)}
=
\frac{\exp(x_i)\exp(-c)}{\sum_j \exp(x_j)\exp(-c)}
=
\frac{\exp(x_i)}{\sum_j \exp(x_j)}.
\]

The same factor, \(\exp(-c)\), appears on the top and bottom, so it cancels. If \(c\) is the largest score, every shifted score is zero or negative. Its exponential is therefore between zero and one. This prevents the exponential step from overflowing.

The final sum still needs enough range and accuracy. That is why the code below uses FP32 for the subtraction, exponentials, and sum.

The phrase **shift invariant** gives this simple fact a name: adding or subtracting the same value from every score changes the scores but does not change the softmax probabilities. An **invariant** is just something that stays true while other parts of a calculation change.

## Why number formats matter

Computers have limited space for each number. A floating-point format stores a sign, the important digits, and a scale using a fixed number of bits. This creates two limits:

- **Range** means the smallest and largest values the format can hold. A value that is too large becomes infinity. A very small value may become zero.
- **Precision** means how much detail the format can keep. Two values that are close together may be rounded to the same stored number.

FP16 uses fewer bits than FP32. It needs less memory and is often faster, but its range is much smaller. That is why `exp(12)` fails even though 12 itself looks harmless. The [IEEE 754 standard](https://standards.ieee.org/ieee/754/6210/) defines floating-point numbers, infinities, and NaNs. [NVIDIA's mixed-precision guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) explains FP16's limits and why some steps should still use FP32.

Subtracting the maximum prevents overflow inside softmax. It cannot repair damage that happened earlier. If the scores already became infinite, or if rounding already made different scores equal, converting them to FP32 later cannot recover the lost information.

## Masking creates a different failure

Softmax has another edge case that subtraction cannot solve.

A **transformer** is the model design behind many modern language and vision models. Transformers use **attention** to decide which input positions matter for each output. Some positions must be ignored. For example, padding added to make inputs the same length should not affect the answer. Marking a position as unavailable is called **masking**.

Code often replaces a masked score with negative infinity. Its exponential becomes zero, so it receives zero probability. A **row** here means one group of scores that compete with each other, such as all positions that one word is allowed to attend to.

That works while each row contains at least one valid position. If every position is masked, the row becomes:

\[
[-\infty,-\infty,-\infty].
\]

Its largest value is still negative infinity. Subtracting that value causes `-∞ - (-∞)`, which is NaN. There is also a deeper problem: probabilities cannot be assigned when there are no valid choices.

The system must decide what an empty row means. It can reject the input, return all zeros with a separate flag that says the row is invalid, or follow another clearly documented rule. It should not return equal probabilities. That would pretend the blocked choices were valid.

## A runnable diagnostic

The following NumPy program shows both failures. **NumPy** is a Python library for working with arrays of numbers. The program compares the safe result with a more accurate calculation and gives an all-masked row a clear result.

Three code terms are useful before reading it:

- **dtype**, short for data type, says how a number is stored, such as FP16, FP32, or FP64.
- **axis** tells NumPy which direction contains the scores that compete with each other. `keepdims=True` keeps the output in a shape that NumPy can subtract from every score in the row.
- An **FP64 reference** repeats the calculation with a wider number format. It is not exact mathematics, but it is accurate enough to find the errors shown here.

```python
import numpy as np


def naive_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        z = np.exp(x)
        return z / z.sum(axis=axis, keepdims=True)


def stable_softmax(
    x: np.ndarray,
    axis: int = -1,
    compute_dtype: np.dtype = np.float32,
) -> np.ndarray:
    work = x.astype(compute_dtype)
    shifted = work - work.max(axis=axis, keepdims=True)
    numer = np.exp(shifted)
    return numer / numer.sum(axis=axis, keepdims=True)


def checked_masked_softmax(
    logits: np.ndarray,
    valid: np.ndarray,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    if logits.shape != valid.shape:
        raise ValueError("logits and valid must have identical shapes")
    if valid.dtype != np.bool_:
        raise TypeError("valid must be boolean")
    if not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("logits must use a floating-point dtype")

    work = logits.astype(np.float32)
    row_valid = valid.any(axis=axis, keepdims=True)
    masked = np.where(valid, work, -np.inf)
    row_max = masked.max(axis=axis, keepdims=True)

    # Empty rows need an explicit policy, not -inf - (-inf).
    safe_max = np.where(row_valid, row_max, 0.0)
    numer = np.exp(masked - safe_max)
    numer = np.where(valid, numer, 0.0)
    denom = numer.sum(axis=axis, keepdims=True)
    probs = np.divide(numer, denom, out=np.zeros_like(numer), where=row_valid)
    return probs, np.squeeze(row_valid, axis=axis)


def main() -> None:
    x = np.array([[12.0, 11.0, 10.0]], dtype=np.float16)
    naive = naive_softmax(x)
    stable = stable_softmax(x)
    reference = stable_softmax(x, compute_dtype=np.float64)

    assert not np.isfinite(naive).all()
    np.testing.assert_allclose(stable, reference, atol=1e-7, rtol=1e-6)

    logits = np.array([[3.0, 1.0], [5.0, 4.0]], dtype=np.float16)
    valid = np.array([[True, False], [False, False]], dtype=bool)
    probs, row_valid = checked_masked_softmax(logits, valid)

    assert row_valid.tolist() == [True, False]
    np.testing.assert_array_equal(probs[1], np.zeros_like(probs[1]))

    try:
        checked_masked_softmax(
            np.array([[3, 2]], dtype=np.int32),
            np.array([[True, True]], dtype=bool),
        )
    except TypeError:
        pass
    else:
        raise AssertionError("integer logits must be rejected")

    print("naive FP16:", naive)
    print("stable FP32:", stable)
    print("masked probabilities:", probs)
    print("row validity:", row_valid)


if __name__ == "__main__":
    main()
```

Save the code as `stable_softmax_diagnostic.py`. With Python 3.10 or newer installed, run:

```bash
python -m pip install numpy
python stable_softmax_diagnostic.py
```

The first result should contain NaN. The stable result should match the FP64 result. The row with no valid choices should return zeros and `False` rather than fake probabilities. The publication package also includes this script as a separate file.

## What to use in real systems

A machine-learning library such as PyTorch provides tested softmax operations. PyTorch calls its arrays **tensors**. `log_softmax` calculates the logarithm of each softmax probability without first creating fragile probability values. A **logarithm** reverses an exponential. [Cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) is a common training measure that compares the model's scores with the correct answer. The framework operation also avoids unnecessary intermediate steps.

The safe default is straightforward:

1. Pass raw scores to the library's `softmax`, `log_softmax`, or cross-entropy operation. Do not build the operation by joining `exp`, `sum`, division, and `log` by hand.
2. Use FP32 for the maximum, subtraction, exponentials, and sum if those steps would otherwise run entirely in FP16. A tested faster implementation can be used if it gives accurate results.
3. Define what a fully masked row means and test that case explicitly.
4. Compare difficult rows with an FP64 reference during testing.
5. Track NaNs, infinities, probability totals, and how many probabilities were rounded to zero.

The PyTorch documentation explains why separate `log(softmax(x))` steps are less safe and why two mathematically equal floating-point calculations can give slightly different stored results ([PyTorch `softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html), [PyTorch `log_softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html), [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)).

## When optimized kernels enter the picture

A **graphics processing unit (GPU) kernel** is a small program that runs directly on a GPU. A **fused kernel** joins several steps into one program. This can avoid moving temporary results to and from slower GPU memory.

[FlashAttention](https://arxiv.org/abs/2205.14135), for example, processes attention in smaller blocks. It keeps a running maximum and sum instead of storing the full table of attention scores. The same safe idea can update softmax one block at a time, as described in [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867).

These kernels can be faster and use less memory. But two calculations can be equal on paper and still differ slightly on a computer because changing the order of operations changes rounding.

**Profiling** measures where a program spends its time and memory. **Latency** is how long one request takes. **Throughput** is how many requests or tokens the system handles in a fixed time. A **microbenchmark** tests one small operation by itself, so it may not show whether the full model becomes faster.

A specialized kernel should replace the framework default only when:

- measurements show that softmax limits request time, work per second, or the maximum input length;
- very large and small scores, masks, outputs, and **gradients** agree closely enough with the FP64 test result; gradients are the signals used to update the model during training;
- model quality and training progress remain within limits chosen before the test; and
- the end-to-end improvement survives outside a microbenchmark.

Undo the change if it creates NaNs or infinities, treats masks incorrectly, differs too much from the reference, or makes model quality worse.

## Failure patterns to recognize

### The loss suddenly becomes NaN

The **loss** is a number that measures how wrong the model is during training. Training tries to make it smaller. If the loss becomes NaN, find the first array that contains NaN or infinity instead of looking only at the final loss. Record the smallest and largest score, number format, number of valid positions, and probability total.

Common causes include calculating exponentials directly, scores that became infinite before softmax, using too little precision inside softmax, and rows with no valid choices.

### Outputs are finite, but learning becomes worse

Values can be finite and still be wrong enough to hurt training. A tiny probability may round to zero and remove a useful update signal. Compare FP32 with FP64, count zero probabilities, and check whether the most likely choices change.

Also test **calibration**, which asks whether confidence matches reality. For example, predictions made with 80% confidence should be correct about 80% of the time across many examples.

## What changes when many teams and GPUs are involved

At Staff or Principal engineer scope, this cannot remain a private detail inside one model. The organization needs a shared rule that states:

- which number formats are supported;
- what every mask case means, including an empty row;
- how far results and gradients may differ from the FP64 reference;
- what a faster kernel must pass before release;
- which warning signs are tracked; and
- when the system must return to the older implementation.

The allowed difference from the reference is called an **error budget**.

Large models often train on many GPUs at once. Each running copy of the training program is often called a **worker**. Workers combine their gradients with an operation such as `all_reduce`, which gathers values across workers and gives the combined result back to each worker. One NaN can therefore spread from one worker to the whole training job ([PyTorch distributed operations](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)).

The training platform should count NaNs, infinities, empty rows, and unusual score ranges automatically. When a problem appears, it should save enough detail to reproduce the bad row. Model teams should not have to rebuild these checks for every project.

## Conclusion

Softmax cares about the gaps between scores, not their absolute size. Subtracting the largest score keeps those gaps unchanged and keeps every exponential at one or below.

The practical rule is simple: use tested softmax operations, perform the sensitive steps in FP32, and decide what an all-masked row means. Use a faster version only after tests show that its outputs, gradients, masks, speed, and model quality stay within clear limits.
