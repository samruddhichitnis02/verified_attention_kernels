# verified_attention_kernels

A small pipeline that turns an attention description into a GPU kernel, checks the answer against a trusted PyTorch version, tunes the kernel on a real GPU, and compares it fairly with PyTorch and FlashInfer.

> **Status: work in progress.** Nothing in this README is claimed as done until it is ticked in the [Roadmap](#roadmap).

---

## What is this, in plain words?

Chatbots are slow and expensive to run. A big reason is one step inside them called **attention**, where the model decides which earlier words matter for the next one.

A GPU can do attention very fast, but only if someone writes a careful small program for it. That program is called a **kernel**. Different kinds of attention need different kernels, and a kernel can be fast and still give wrong answers.

This project builds the whole loop around that problem:

1. Describe the attention you want in a few lines.
2. Generate a GPU kernel for it.
3. Prove the answer is correct.
4. Find the fastest settings on a real GPU.
5. Compare the result fairly with existing tools.

The rule of the project: **a kernel is only accepted if it is correct first and faster second.**

---

## How it works

```
Attention spec
      |
      v
PyTorch reference   (slow, simple, trusted answer key)
      |
      v
Kernel generator    (Triton, plus one hand written CUDA C++ kernel)
      |
      v
Correctness checker (random and edge case tests)
      |
      v
Autotuner           (tries block sizes, warps, stages, keeps the fastest correct one)
      |
      v
Fair benchmark      (same workload as PyTorch and FlashInfer)
      |
      v
Profile and report  (Nsight, and a written explanation of why it is fast)
```

### Example spec

```
mask:            causal
window_size:     256
dtype:           fp16
head_dim:        128
gqa_groups:      8
```

---

## What is supported in version 1

The scope is small on purpose. A few cases done properly beat many cases done badly.

| Case | What it means |
|---|---|
| Causal prefill attention | Each word looks only at earlier words, processing a whole prompt at once |
| Sliding window causal attention | Each word looks only at the last N words |
| GQA decode attention | Generating one token at a time, reading a saved key value cache where several query heads share one key value head |

---

## How a kernel gets accepted

A generated or tuned kernel must pass all of these:

- **Matches the PyTorch reference** within a stated maximum absolute error and relative error
- **Passes random tests** over batch sizes, sequence lengths, head sizes, masks, and data types (FP16, BF16, FP32)
- **Passes edge cases**: sizes that do not divide evenly into tiles, very short and very long sequences, large and small input values, empty or tiny windows
- **Is actually faster** on the tested workload, measured with the rules below

If any check fails, the kernel is thrown away.

---

## How benchmarks are done

Comparisons are only meaningful when the setup is identical.

| Must match | Details |
|---|---|
| Operation | Prefill or decode, never mixed |
| Mask | Same causal or window mask |
| Shapes | Same batch, sequence length, heads, head dimension |
| Data type | Same FP16, BF16, or FP32 |
| Output definition | Same numerical meaning |
| Timing | CUDA events, warm up runs, synchronization, repeated trials, median latency |
| Memory | Peak memory and extra workspace reported |
| Accuracy | Max and relative error reported next to every speed number |

FlashInfer and PyTorch FlexAttention are built for somewhat different workloads, so only matching operations are compared. Where they do not match, that is written down instead of hidden.

---

## Tech

| Area | What is used |
|---|---|
| Language | Python, C/C++ |
| Framework | PyTorch |
| Kernel writing | Triton, CUDA C++ (as a PyTorch extension) |
| Attention method | Tiled, memory efficient attention in the style of Flash Attention |
| Compared against | PyTorch attention, PyTorch FlexAttention, FlashInfer |
| Profiling | Nsight Compute, Nsight Systems |
| Tuning | Block sizes, number of warps, pipeline stages |

---

## Hardware note

All results come from a single NVIDIA GPU. The exact GPU, driver, CUDA, PyTorch, and Triton versions are listed next to every result. Kernel settings that are best on one GPU are often not best on another, because registers, caches, and shared memory differ. Retuning for other GPUs is future work.

---

## Roadmap

- [ ] Repo setup and project description
- [ ] Attention spec format
- [ ] PyTorch reference for all three cases
- [ ] Triton kernel for causal prefill attention
- [ ] Correctness checker with random and edge case tests
- [ ] Timing harness (CUDA events, warm up, median)
- [ ] Autotuner for the causal kernel
- [ ] Sliding window causal kernel
- [ ] GQA decode kernel
- [ ] One hand written CUDA C++ kernel for a simple case
- [ ] Fair benchmark against PyTorch and FlashInfer
- [ ] Nsight profile and a written bottleneck explanation
- [ ] Optional: AI helper that proposes kernel changes, accepted only if tests and benchmarks pass
- [ ] Optional: small contribution to an open source kernel library

---

## Planned repo layout

```
verified_attention_kernels/
  specs/          attention spec files
  reference/      trusted PyTorch implementations
  kernels/
    triton/       Triton kernels
    cuda/         CUDA C++ kernels and PyTorch extension
  tuner/          autotuning code
  tests/          correctness and edge case tests
  bench/          timing and comparison scripts
  profiles/       Nsight notes and reports
  README.md
```

---

## What this project is not

- It is not a system that magically writes every possible GPU kernel.
- It is not a replacement for FlashInfer or cuDNN. Those are far more complete.
- It does not claim results on any GPU other than the one used for testing.

It is a scoped, honest pipeline that shows how to go from a spec to a checked, tuned, and explained kernel.

---

## Setup

Setup steps will be added once the first kernel runs.

---

## References

- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- Triton: https://github.com/triton-lang/triton
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- PyTorch FlexAttention: see the PyTorch documentation and blog
