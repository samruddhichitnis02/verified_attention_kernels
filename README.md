# Verified Attention Kernels

**A correctness-first GPU kernel pipeline for attention workloads.**

Verified Attention Kernels focuses on one practical problem in deep learning systems:

> How do we make attention faster without accepting a kernel that produces an incorrect answer?

The project takes a structured attention description, matches it with a trusted PyTorch reference, runs a Triton or CUDA implementation, validates the numerical output, searches for efficient GPU launch parameters, and compares equivalent workloads with established implementations.

The project is intentionally focused on a small family of attention workloads. The goal is not to claim that one program generates every possible GPU kernel. The goal is to build a clear, testable, and explainable path from an attention description to a GPU implementation.

> **Correctness comes first. Performance is considered only after correctness passes.**

---

## Why attention matters

Attention is a central operation in modern language models.

Imagine a chatbot receives:

> “The capital of France is ___”

Before predicting the next word, the model looks at the words in the prompt and decides which ones are relevant. The word “France” matters more for this prediction than unrelated words.

This process is called **attention**.

For a short prompt, attention is manageable. For a long prompt, the model compares many tokens with one another. This creates a large amount of:

- matrix multiplication;
- memory movement;
- softmax computation;
- intermediate data;
- GPU synchronization;
- kernel launch overhead.

A GPU can execute this work quickly, but only when the computation is organized carefully around the GPU hardware.

A specialized GPU function for one operation is called a **kernel**.

Different attention workloads need different kernels because their computation and memory access patterns are different. A kernel that is fast for one shape may be slow for another. A kernel that is fast but numerically wrong is unusable.

This project treats correctness and performance as equally important.

---

## Attention in simple terms

Attention uses three tensors:

- **Query:** what the current token is looking for;
- **Key:** the information associated with every token;
- **Value:** the content returned when a key is considered relevant.

A simplified version of attention is:

```text
scores = Query × Key^T
weights = softmax(scores)
output = weights × Value
```

The score measures how relevant each key is to a query. Softmax converts the scores into weights. The weighted values form the output passed to the next layer.

The mathematical expression is short, but an efficient GPU implementation must also handle:

- global memory loads and stores;
- tile reuse;
- stable softmax;
- masking;
- thread and warp organization;
- register usage;
- shared memory;
- tensor shapes that do not divide evenly;
- reduced precision such as FP16 and BF16.

That gap between a simple mathematical operation and an efficient hardware implementation is the main focus of this project.

---

## What is a GPU kernel?

A kernel is a function executed by many GPU threads in parallel.

A high-level PyTorch operation may look like this:

```python
output = torch.matmul(query, key.transpose(-2, -1))
```

PyTorch selects and launches lower-level GPU kernels to perform the operation.

This project studies the kernel level. It controls decisions such as:

- how queries and keys are divided into tiles;
- which threads load each part of the tensors;
- how data is reused;
- how softmax is computed;
- how many warps participate;
- how much shared memory and register space is used;
- whether large intermediate tensors are written to global memory.

A kernel is not considered successful merely because it compiles. It must match the trusted reference within defined numerical tolerances and must be measured under a reproducible benchmark setup.

---

## Project objective

The objective is to build a small compiler-like and autotuning pipeline:

```text
attention description
        |
        v
trusted PyTorch reference
        |
        v
Triton or CUDA kernel
        |
        v
correctness validation
        |
        v
autotuning
        |
        v
fair benchmark
        |
        v
Nsight performance analysis
```

The pipeline connects:

- a structured workload description;
- a deep learning framework;
- GPU kernel implementations;
- numerical validation;
- configuration search;
- runtime measurement;
- hardware-level analysis.

This makes the project more than an isolated custom kernel. It represents the surrounding library and runtime workflow required to use a kernel reliably.

---

## Attention specification

The workload is described using a small structured specification.

Example:

```yaml
operation: causal_prefill
mask: causal
window_size: null
dtype: fp16
head_dim: 128
num_query_heads: 32
num_kv_heads: 32
batch_size: 2
sequence_length: 2048
```

The fields describe:

| Field | Meaning |
|---|---|
| operation | The attention workload, such as prefill or decode |
| mask | Which query-key pairs are allowed |
| window_size | The local attention range when windowed attention is used |
| dtype | FP16, BF16, or FP32 |
| head_dim | The size of each attention head |
| num_query_heads | Number of query heads |
| num_kv_heads | Number of key-value heads |
| batch_size | Number of examples processed together |
| sequence_length | Number of tokens in the sequence |

The specification gives the system explicit information instead of asking the kernel generator to infer every detail from free-form text.

---

## Supported workload families

The project focuses on three workload families.

| Workload | Simple meaning | Main systems challenge |
|---|---|---|
| Causal prefill attention | Process a complete prompt while each token sees only allowed earlier tokens | Tiling, masking, online softmax, and memory reuse |
| Sliding-window causal attention | Each token sees only a fixed number of recent tokens | Efficient local masking and avoiding unnecessary work |
| GQA decode attention | Generate one token at a time while reading a saved key-value cache | Cache access, head mapping, and decode-time memory behavior |

### Causal prefill attention

During prefill, the model processes many prompt tokens at once.

With causal attention, a token cannot use information from the future. Token 5 can use tokens 1 through 5, but it cannot use token 6.

The kernel handles:

- the causal mask;
- tiled query and key blocks;
- stable softmax;
- efficient data reuse;
- sequence lengths that do not align with tile sizes.

Prefill is the first workload because it provides a clear starting point for understanding attention tiling and correctness.

### Sliding-window causal attention

In sliding-window attention, each token attends only to a local range.

For a window size of 256, a token attends to at most the most recent 256 permitted tokens.

This reduces unnecessary work for long contexts, but it creates additional boundary cases:

- the valid key range changes for each query row;
- some blocks are completely outside the window;
- some blocks are only partially valid;
- masked values must not affect softmax;
- unnecessary memory loads reduce the value of the local window.

### GQA decode attention

During autoregressive generation, the model generates one token at a time.

The keys and values from previous tokens are kept in a **key-value cache**. The model reads this cache instead of recomputing the entire previous sequence at every decoding step.

Grouped Query Attention reduces the number of key-value heads. Several query heads share one key-value head.

Example:

```text
32 query heads
8 key-value heads
4 query heads share each key-value head
```

GQA decode is a separate kernel family because it introduces:

- key-value cache reads;
- query-to-key-value head mapping;
- decode-specific memory access;
- different batch and sequence behavior;
- different bottlenecks from prefill attention.

---

## System components

### 1. Reference implementation

The PyTorch reference is clear and simple rather than highly optimized.

It provides the expected numerical result for the same input used by the candidate kernel.

The reference is used for:

- output comparison;
- randomized tests;
- edge-case tests;
- numerical tolerance checks;
- regression checks after kernel changes.

The reference is the correctness oracle, not the performance target.

### 2. Kernel implementation

The candidate kernel is written in Triton or CUDA C++.

The first kernel uses a restricted workload so the implementation is easier to inspect and debug. The scope includes clearly defined:

- tensor layouts;
- supported data types;
- head dimensions;
- masks;
- sequence lengths;
- batch behavior.

The kernel interface follows the same input and output definition as the reference implementation.

### 3. Correctness gate

Every candidate passes through a correctness gate before performance selection.

The gate compares the candidate output with the reference output using:

- maximum absolute error;
- maximum relative error;
- data-type-specific tolerances;
- randomized inputs;
- boundary and edge-case inputs.

A candidate that fails the gate is rejected from the tuning and benchmark results.

### 4. CUDA event benchmark

GPU operations execute asynchronously. Ordinary Python timing can include CPU overhead or measure the wrong interval.

The benchmark uses:

- warm-up executions;
- CUDA events;
- explicit synchronization;
- repeated trials;
- median latency;
- timing variation;
- separate compilation and steady-state measurements.

The timing harness measures the kernel execution that matters for the workload instead of mixing compilation time with steady-state latency.

### 5. Autotuner

A GPU kernel can run at different speeds with different launch configurations.

The autotuner searches parameters such as:

- query tile size;
- key tile size;
- number of warps;
- pipeline stages;
- block layout;
- split-K strategy where applicable;
- legal launch parameters for the target GPU.

Each configuration passes correctness before timing. The selected configuration is the fastest configuration that satisfies the numerical checks for the specified workload.

### 6. Benchmark comparison

The selected kernel is compared against equivalent PyTorch and FlashInfer operations.

The comparison includes:

- latency;
- numerical error;
- data type;
- tensor shapes;
- masks;
- operation type;
- memory use where measurable;
- GPU and software environment.

The comparison does not mix prefill with decode or compare different masks as if they were the same workload.

### 7. Nsight analysis

Benchmark timing answers:

> How fast is the implementation?

Nsight Systems and Nsight Compute help answer:

> Why does the implementation perform that way?

The analysis covers:

- global memory traffic;
- memory coalescing;
- shared memory;
- register usage;
- occupancy;
- arithmetic intensity;
- floating point and tensor core utilization;
- warp stalls;
- synchronization;
- kernel launch overhead;
- compute-bound versus memory-bound behavior.

---

## Kernel design concepts

### Tiling

The full attention matrix is not processed as one enormous object. The kernel processes smaller blocks called tiles.

A tile contains a block of query rows and a block of key columns. Processing tiles allows the kernel to reuse data and avoid materializing the entire attention matrix.

Tile size affects:

- data reuse;
- register pressure;
- shared-memory usage;
- occupancy;
- boundary handling;
- instruction count.

### Online softmax

A simple implementation may create a large score matrix and apply softmax afterward.

A memory-efficient attention kernel updates the softmax statistics while processing tiles. This reduces intermediate global-memory traffic.

The kernel maintains running softmax values carefully so that the result remains numerically stable.

### Masking

The kernel must prevent invalid query-key pairs from affecting the result.

For causal attention, future tokens are masked.

For sliding-window attention, tokens outside the local window are masked.

Masking affects both correctness and performance. Incorrect masking changes the model output. Poor masking can make the kernel perform work on values that do not contribute to the result.

### Memory movement

Many attention optimizations reduce unnecessary memory movement.

The implementation may improve data movement by:

- using coalesced loads;
- reusing query, key, and value tiles;
- reducing global-memory writes;
- avoiding large intermediate tensors;
- using shared memory carefully;
- keeping frequently reused values in registers when practical.

A custom kernel is not automatically faster. The benchmark and profiler provide the evidence.

---

## Correctness methodology

### Numerical comparison

The test harness compares the candidate output with the reference output.

The main measurements are:

- maximum absolute error;
- maximum relative error;
- mean absolute error as an additional diagnostic.

Tolerances depend on the operation and data type. FP16 and BF16 can show larger numerical differences than FP32 because the computations use lower precision.

A candidate passes only when it satisfies the defined tolerance policy.

### Randomized testing

The tests vary:

- batch size;
- sequence length;
- number of query heads;
- number of key-value heads;
- head dimension;
- mask type;
- input magnitude;
- data type;
- dimensions that do not align with tile sizes.

A kernel that succeeds for one convenient shape is not enough. Randomized tests help expose incorrect indexing, masking, boundary handling, and numerical behavior.

### Edge-case testing

The test suite includes:

- sequence lengths smaller than the tile size;
- sequence lengths that do not divide evenly into tiles;
- head dimensions that do not match the normal tile width;
- very short sequences;
- longer sequences;
- single and multiple batches;
- small and large input values;
- causal boundary rows;
- sliding-window boundary rows;
- GQA head mappings;
- tiny valid windows where the workload permits them.

### Meaning of verified

“Verified” means that a candidate passes the defined numerical validation suite.

It does not mean that a formal mathematical proof covers every possible input.

The validation claim is tied to:

- tested shapes;
- tested data types;
- tested masks;
- stated tolerances;
- tested edge cases.

---

## Autotuning methodology

Autotuning measures multiple legal kernel configurations and chooses the fastest correct configuration for a specific workload.

For each attention specification, the tuner:

1. Creates the legal configuration space.
2. Compiles or launches each candidate.
3. Runs the correctness gate.
4. Removes configurations that fail.
5. Warms up the remaining candidates.
6. Measures repeated execution with CUDA events.
7. Calculates median latency and timing variation.
8. Selects the fastest correct configuration.

A configuration tuned for a 2,048-token sequence is not automatically optimal for a 16,384-token sequence. Kernel performance depends on the workload shape and the target GPU.

---

## Benchmarking methodology

### Identical workload requirements

The custom kernel and comparison implementation use the same:

- operation;
- batch size;
- sequence length;
- number of query heads;
- number of key-value heads;
- head dimension;
- mask;
- data type;
- scaling rule;
- output definition;
- device.

### Timing procedure

The benchmark follows this process:

1. Compile or warm up the implementation.
2. Run several warm-up executions.
3. Start CUDA event timing.
4. Execute the same operation repeatedly.
5. Synchronize at the measurement boundaries.
6. Calculate median latency and timing variation.
7. Report compilation time separately from steady-state execution.

### PyTorch comparisons

PyTorch provides both a familiar framework implementation and the numerical reference.

The comparison can include:

- the clear PyTorch reference;
- PyTorch scaled dot product attention;
- PyTorch FlexAttention when its behavior matches the workload.

The exact PyTorch path appears next to each result.

### FlashInfer comparisons

FlashInfer targets efficient attention and inference workloads, but not every FlashInfer operation matches every custom kernel.

A FlashInfer comparison is valid only when:

- the operation is the same;
- the mask is the same;
- the tensor layout is equivalent;
- the data type is the same;
- the output definition is the same;
- the benchmark includes the same amount of work.

A non-equivalent operation is not presented as a direct performance comparison.

---

## Profiling methodology

### Nsight Systems

Nsight Systems examines the broader execution timeline:

- kernel launch sequence;
- CPU to GPU interaction;
- synchronization;
- gaps between kernels;
- repeated operations;
- overall execution flow.

### Nsight Compute

Nsight Compute examines individual kernels:

- achieved occupancy;
- memory throughput;
- global-memory efficiency;
- shared-memory behavior;
- register usage;
- instruction throughput;
- warp stalls;
- arithmetic intensity;
- compute and memory utilization.

The purpose of profiling is to connect a performance result to a hardware-level explanation.

For example:

```text
Measured result:
The custom kernel has lower latency.

Implementation explanation:
The kernel avoids writing a complete attention score matrix to global memory.

Profiler evidence:
The profile shows lower global-memory traffic and fewer intermediate writes.
```

A custom kernel is not described as optimized merely because it is handwritten. The performance claim depends on timing and profiling evidence.

---

## Triton and CUDA C++ roles

### Triton

Triton provides a productive way to explore GPU kernel designs.

It exposes important decisions such as:

- tile shapes;
- memory layouts;
- warp count;
- pipeline stages;
- masking;
- fused operations;
- autotuning parameters.

Triton provides a shorter development loop while still requiring GPU-level reasoning.

### CUDA C++

CUDA C++ provides a lower-level implementation for a simple supported workload.

The CUDA kernel integrates with PyTorch through a C++ or CUDA extension. This covers:

- C++ and CUDA integration;
- thread and block organization;
- device memory;
- synchronization;
- compilation;
- framework integration.

The project measures Triton and CUDA C++ on the same defined workload rather than assuming one is always faster.

---

## Repository structure

```text
verified_attention_kernels/
├── specs/
│   ├── causal_prefill.yaml
│   ├── sliding_window.yaml
│   └── gqa_decode.yaml
│
├── reference/
│   ├── causal_prefill.py
│   ├── sliding_window.py
│   └── gqa_decode.py
│
├── kernels/
│   ├── triton/
│   │   ├── causal_prefill.py
│   │   ├── sliding_window.py
│   │   └── gqa_decode.py
│   └── cuda/
│       ├── attention_kernel.cu
│       ├── attention_binding.cpp
│       └── setup.py
│
├── tests/
│   ├── test_correctness.py
│   ├── test_edge_cases.py
│   └── test_tolerances.py
│
├── tuner/
│   ├── search_space.py
│   ├── autotune.py
│   └── results.py
│
├── bench/
│   ├── benchmark.py
│   ├── compare_pytorch.py
│   ├── compare_flashinfer.py
│   └── environment.py
│
├── profiles/
│   ├── nsight_systems/
│   ├── nsight_compute/
│   └── analysis.md
│
├── results/
│   ├── correctness/
│   ├── benchmarks/
│   └── metadata/
│
└── README.md
```

---

## Technology stack

| Area | Technology |
|---|---|
| Language | Python and C/C++ |
| Deep learning framework | PyTorch |
| GPU kernel development | Triton and CUDA C++ |
| Kernel integration | PyTorch C++/CUDA extension |
| Attention workloads | Causal prefill, sliding-window attention, GQA decode |
| Comparisons | PyTorch, PyTorch FlexAttention, FlashInfer |
| Timing | CUDA events and synchronized repeated trials |
| Profiling | Nsight Systems and Nsight Compute |
| Hardware focus | NVIDIA CUDA GPUs |

---

## Project summary

Verified Attention Kernels is a correctness-first GPU systems project centered on attention optimization.

It connects:

- PyTorch reference implementations;
- Triton kernels;
- CUDA C++ extensions;
- randomized numerical validation;
- autotuning;
- synchronized GPU benchmarking;
- FlashInfer comparisons;
- Nsight Systems and Nsight Compute analysis.

The important engineering idea is the order of operations:

1. Define the workload.
2. Build a trusted reference.
3. Implement the kernel.
4. Validate numerical correctness.
5. Tune legal configurations.
6. Benchmark equivalent implementations.
7. Explain the performance using profiling data.

A concise description of the project is:

> I am building a correctness-first autotuning pipeline for attention kernels. A structured workload specification selects a trusted PyTorch reference and a Triton or CUDA implementation. The candidate passes randomized and edge-case numerical tests before it enters the tuning search. The tuner measures tile, warp, and pipeline configurations using synchronized CUDA events and selects the fastest correct configuration for the target workload. The result is compared fairly with PyTorch and FlashInfer, while Nsight explains whether performance is limited by memory movement, computation, occupancy, synchronization, or launch overhead. An optional AI helper proposes changes, but automated correctness and performance checks decide whether they are accepted.

---

## References

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [Triton](https://github.com/triton-lang/triton)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)
