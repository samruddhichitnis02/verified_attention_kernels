# Verified Attention Kernels

**A correctness-first pipeline for generating, validating, tuning, and analyzing GPU attention kernels.**

This project is being developed for deep learning systems and GPU kernel engineering. It focuses on a small but important problem:

> How can we create a faster attention kernel without accepting a kernel that produces the wrong answer?

The project takes a structured attention description, creates a trusted PyTorch reference, runs a Triton or CUDA implementation, validates numerical correctness, searches for faster GPU configurations, and compares equivalent workloads against established implementations.

The project is intentionally scoped. It does not claim to generate every possible GPU kernel. It focuses on a constrained family of attention workloads so that correctness, performance, and the reason for every optimization can be explained clearly.

> **Correctness comes first. Performance is measured only after correctness passes.**

---

## Project status

This repository documents the complete design and staged implementation plan.

The project is being built in the following order:

1. Attention specification
2. Trusted PyTorch reference
3. Restricted causal Triton kernel
4. Correctness and edge case tests
5. CUDA event benchmark harness
6. Autotuning
7. Sliding window attention
8. GQA decode attention
9. CUDA C++ implementation
10. Nsight profiling and performance analysis
11. Fair comparisons with PyTorch and FlashInfer
12. Optional AI assisted optimization

Benchmark numbers, speedups, and profiling claims will be added only after they are measured reproducibly on the documented hardware and software environment.

---

## Why this problem matters

Modern language models spend a large amount of time in repeated tensor operations. Attention is one of the most important of these operations.

Suppose a chatbot receives:

> “The capital of France is ___”

Before predicting the next word, the model examines the words in the prompt and decides which ones are relevant. The word “France” is more important for this prediction than unrelated words elsewhere in the prompt.

This process is called **attention**.

For a short prompt, attention is manageable. For a long prompt, the model may need to compare many tokens with many other tokens. That creates substantial:

- matrix multiplication work;
- memory traffic;
- softmax work;
- GPU synchronization;
- intermediate storage;
- kernel launch overhead.

A GPU can perform this work efficiently, but the implementation must be designed around the GPU's memory hierarchy and execution model.

That specialized GPU program is called a **kernel**.

The challenge is that attention is not one fixed operation. Its behavior changes depending on:

- whether the model is processing a complete prompt or generating one token;
- whether attention is causal or local;
- sequence length;
- head dimension;
- data type;
- grouped query attention;
- key value cache layout;
- batch size;
- GPU architecture.

A kernel that is fast for one shape may be slow for another shape. A kernel that is fast may also be numerically wrong. This project treats both problems as first class engineering requirements.

---

## The project in one sentence

> A small compiler-like pipeline converts an attention specification into a GPU kernel, validates it against a trusted PyTorch reference, autotunes its launch configuration, and explains its performance using reproducible benchmarks and NVIDIA profiling tools.

---

## What the system takes in and produces

### Input

A small attention specification:

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

### Processing

The system:

1. Parses the specification.
2. Selects the matching reference implementation.
3. Selects or generates the matching GPU kernel.
4. Runs numerical correctness tests.
5. Searches valid kernel configurations.
6. Measures the fastest correct configuration.
7. Compares it with equivalent PyTorch and FlashInfer implementations.
8. Stores the configuration, environment, correctness results, timing results, and profiling notes.

### Output

For every tested workload, the project should produce:

- the attention specification;
- the selected kernel implementation;
- the selected tuning configuration;
- maximum absolute error;
- relative error;
- latency statistics;
- memory measurements where available;
- GPU and software environment metadata;
- profiling notes explaining the bottleneck;
- a reproducible command for rerunning the experiment.

---

## How attention works in simple terms

Attention uses three representations:

- **Query:** what the current token is looking for;
- **Key:** what information each previous token contains;
- **Value:** the information that is returned when a key is considered relevant.

A simplified attention operation is:

```text
scores = Query × Key^T
weights = softmax(scores)
output = weights × Value
```

The score tells the model how relevant each token is. Softmax converts the scores into weights. The weighted values create the output used by the next layer.

The formula is simple to write, but the efficient GPU implementation is difficult because it must:

- load data from global memory efficiently;
- reuse values instead of loading them repeatedly;
- avoid storing unnecessarily large intermediate matrices;
- perform softmax stably;
- handle masks correctly;
- use threads and warps efficiently;
- work for different tensor shapes.

This is why attention kernels are a useful case study for GPU systems engineering.

---

## What is a kernel?

A kernel is a function executed by many GPU threads.

A simple PyTorch operation may be expressed as:

```python
output = torch.matmul(query, key.transpose(-2, -1))
```

PyTorch then selects and launches one or more lower level GPU kernels.

In this project, the goal is to study what happens underneath that high level operation. We write or generate a specialized kernel that controls important implementation choices such as:

- how queries and keys are divided into tiles;
- which threads load which values;
- how data is reused;
- how softmax is computed;
- how many warps execute the work;
- how much shared memory and register space is used;
- whether intermediate results are written to global memory.

The kernel is not accepted merely because it compiles. It must produce an answer close enough to the trusted reference and must be measured on the target workload.

---

## The complete pipeline

```text
1. Attention specification
          |
          v
2. Trusted PyTorch reference
          |
          v
3. Triton or CUDA kernel
          |
          v
4. Correctness validation
          |
          v
5. CUDA event benchmark
          |
          v
6. Autotuning search
          |
          v
7. Fair comparison
          |
          v
8. Nsight profiling
          |
          v
9. Reproducible report
```

### Step 1: Attention specification

The specification is a small structured description of the workload.

It defines:

- operation type;
- mask type;
- local window size;
- data type;
- head dimension;
- number of query heads;
- number of key value heads;
- batch size;
- sequence length;
- key value cache layout for decode workloads.

The specification prevents the system from relying on vague natural language alone. It gives the compiler-like pipeline explicit information to use.

### Step 2: Trusted PyTorch reference

The project first implements a clear PyTorch version of the same operation.

This implementation is not intended to be the fastest. Its purpose is to provide a readable and trusted answer for a given input.

The reference is used as the numerical oracle for:

- kernel correctness;
- randomized testing;
- edge case testing;
- regression testing;
- tolerance checks after tuning.

The project uses the phrase **trusted reference**, not “mathematical proof.” Passing the reference tests demonstrates empirical correctness for the tested inputs and tolerances.

### Step 3: Triton or CUDA kernel

The candidate implementation performs the same mathematical operation using GPU-specific code.

The first candidate is intentionally restricted:

- one operation;
- one supported data type;
- fixed or limited head dimensions;
- clearly defined tensor layouts;
- limited mask behavior.

The scope expands only after the smaller kernel is correct and benchmarkable.

### Step 4: Correctness validation

The candidate kernel is compared with the PyTorch reference.

The comparison uses:

- maximum absolute error;
- maximum relative error;
- dtype-specific tolerances;
- randomized inputs;
- edge case inputs.

A failed candidate does not proceed to performance tuning.

### Step 5: CUDA event benchmark

GPU operations are asynchronous. Measuring them with ordinary Python wall clock timing can include unrelated CPU behavior or can finish before the GPU has completed its work.

The benchmark harness therefore uses:

- warm up executions;
- CUDA events;
- explicit synchronization;
- repeated trials;
- median latency;
- measurement spread;
- separate compilation and steady state timing where relevant.

The benchmark also records the exact workload and environment.

### Step 6: Autotuning

The same kernel can run at different speeds depending on its configuration.

The autotuner searches combinations such as:

- query tile size;
- key tile size;
- number of warps;
- pipeline stages;
- block layout;
- split K strategy where applicable;
- launch parameters allowed by the target GPU.

Every configuration must pass correctness before it can become the selected configuration.

### Step 7: Fair comparison

The selected kernel is compared against existing implementations only for equivalent operations.

The comparison records:

- latency;
- maximum absolute error;
- relative error;
- memory use where measurable;
- hardware and software versions;
- operation type;
- tensor shapes;
- mask;
- data type.

The project does not present an unfair comparison as a performance win.

### Step 8: Nsight profiling

Benchmark timing answers:

> How fast was it?

Nsight profiling helps answer:

> Why was it that fast or slow?

The analysis examines:

- global memory loads and stores;
- memory coalescing;
- shared memory;
- registers;
- occupancy;
- arithmetic intensity;
- tensor core or floating point utilization;
- synchronization;
- kernel launch overhead;
- pipeline behavior;
- compute bound versus memory bound behavior.

### Step 9: Reproducible report

Every reported result includes enough information for another person to understand and rerun the experiment.

The report records:

- the specification;
- the input shapes;
- the data type;
- the selected configuration;
- correctness tolerances;
- timing method;
- latency statistics;
- memory measurements;
- GPU model;
- driver version;
- CUDA version;
- PyTorch version;
- Triton version;
- FlashInfer version;
- profiling observations.

---

## Supported workload families

The first version focuses on three attention workloads. These workloads cover different inference behaviors and expose different kernel challenges.

| Workload | Plain language meaning | Why it matters |
|---|---|---|
| Causal prefill attention | Process a complete prompt while each token sees only allowed earlier tokens | Represents the prompt processing phase |
| Sliding window causal attention | Each token looks only at the most recent fixed number of tokens | Reduces unnecessary attention work for long contexts |
| GQA decode attention | Generate one token at a time while reading a saved key value cache | Represents autoregressive generation and memory sensitive decode |

### 1. Causal prefill attention

During prefill, the model processes many prompt tokens at once.

With causal attention, a token cannot look at future tokens. For example, token 5 can use tokens 1 through 5, but it cannot use token 6.

The kernel must:

- apply the causal mask;
- compute attention for a complete sequence;
- avoid unnecessary intermediate storage;
- handle tiled query and key blocks;
- maintain numerically stable softmax values.

This is the first kernel family because it provides a clear starting point for correctness and tiling.

### 2. Sliding window causal attention

In sliding window attention, each token looks only at a local region.

For a window size of 256, a token attends to at most the most recent 256 permitted tokens.

This can reduce work for long sequences, but it creates additional implementation issues:

- the valid key range changes by query position;
- some blocks are fully outside the window;
- boundary blocks may be partially valid;
- the kernel must avoid reading or computing unnecessary values;
- irregular masking can reduce efficiency if handled poorly.

### 3. GQA decode attention

During autoregressive generation, the model usually generates one new token at a time.

The keys and values from previous tokens are stored in a **key value cache**, so the model does not recompute the entire sequence for every new token.

Grouped Query Attention reduces the number of key value heads. Multiple query heads share one key value head.

For example:

```text
32 query heads
8 key value heads
4 query heads share each key value head
```

GQA decode is treated as a separate kernel family because it introduces:

- key value cache reads;
- query to key value head mapping;
- decode specific memory access;
- different batch and sequence behavior;
- different performance bottlenecks from prefill.

---

## Kernel design concepts

### Tiling

Instead of processing the full attention matrix at once, the kernel processes smaller blocks called tiles.

A tile may contain:

- a block of query rows;
- a block of key columns;
- the corresponding values.

Tiling helps the GPU reuse data and avoid storing the entire attention matrix.

The tile sizes affect:

- memory reuse;
- register use;
- shared memory;
- occupancy;
- the number of instructions;
- boundary handling.

### Online softmax

A naive attention implementation may create a large score matrix and then apply softmax to it.

A memory efficient kernel can update the softmax statistics while processing blocks. This avoids writing the complete score matrix to global memory.

The implementation must maintain the running maximum and normalization values carefully so that the result remains numerically stable.

### Masking

The kernel must prevent invalid query key pairs from contributing to the output.

For causal attention, future tokens are masked.

For sliding window attention, tokens outside the local range are masked.

Masking is a correctness issue and a performance issue. A correct implementation must produce the same result as the reference. An efficient implementation should avoid doing unnecessary work for fully invalid regions when possible.

### Memory movement

Many attention optimizations are really memory movement optimizations.

The kernel may improve performance by:

- loading data in coalesced patterns;
- reusing tiles;
- reducing global memory writes;
- avoiding materialization of large intermediate tensors;
- using shared memory carefully;
- keeping frequently reused values in registers when practical.

The README will not claim a speedup until profiling and benchmark results show which change helped.

---

## Correctness methodology

The project uses a correctness gate before performance selection.

### Numerical comparison

For each candidate output, the test harness computes:

- maximum absolute error;
- maximum relative error;
- optionally mean absolute error for additional diagnostics.

The tolerances are documented per data type and operation. FP16 and BF16 computations may have larger expected numerical differences than FP32 computations.

A candidate is accepted only when it satisfies the stated tolerance policy.

### Randomized tests

Random tests vary the supported workload dimensions, including:

- batch size;
- sequence length;
- number of query heads;
- number of key value heads;
- head dimension;
- causal versus local masking;
- input magnitude;
- data type;
- dimensions that do not align with tile sizes.

Randomized tests are important because a kernel may work for one convenient shape and fail for another.

### Edge cases

The test suite includes cases such as:

- sequence length smaller than the tile size;
- sequence length not divisible by the tile size;
- head dimension not divisible by the normal tile width;
- very short sequences;
- longer sequences;
- single batch;
- multiple batches;
- small and large input values;
- causal boundary rows;
- sliding window boundary rows;
- GQA head mappings;
- empty or tiny valid windows where the operation permits them.

### What correctness does not mean

Passing the test suite is not a formal mathematical proof for every possible input.

It means:

> The implementation passed the defined numerical validation suite for the tested shapes, dtypes, masks, tolerances, and edge cases.

This distinction is important for honest performance engineering.

---

## Autotuning methodology

Autotuning is the process of measuring multiple legal kernel configurations and choosing the fastest configuration for a specific workload.

The autotuner does not assume that one configuration is best for every input.

For each attention specification, it:

1. Generates the legal configuration space.
2. Compiles or launches each candidate.
3. Runs correctness tests.
4. Discards incorrect candidates.
5. Warms up the accepted candidates.
6. Measures repeated execution time with CUDA events.
7. Records median latency and variation.
8. Selects the fastest correct configuration.
9. Saves the selected configuration with the benchmark result.

The selected configuration is associated with the workload specification. A configuration tuned for a 2048 token sequence may not be best for a 16,384 token sequence.

---

## Benchmarking methodology

### What must be identical

The custom kernel and comparison implementations must use:

- the same operation;
- the same batch size;
- the same sequence length;
- the same number of query heads;
- the same number of key value heads;
- the same head dimension;
- the same mask;
- the same data type;
- the same scaling rule;
- the same output definition;
- the same device.

### Timing procedure

The benchmark uses:

1. Kernel compilation or warm up before measurement.
2. Several warm up executions.
3. CUDA event timing.
4. GPU synchronization at measurement boundaries.
5. Multiple repeated trials.
6. Median latency and measurement spread.
7. Separate reporting of compilation time when applicable.

Compilation time is not mixed with steady state kernel latency.

### Reported values

Each benchmark result should include:

- operation;
- tensor shape;
- data type;
- mask;
- implementation name;
- latency;
- latency statistic;
- maximum absolute error;
- relative error;
- peak memory if available;
- GPU model;
- software versions;
- selected kernel configuration.

### Comparing PyTorch

PyTorch is a useful reference because it represents a normal framework implementation and provides the trusted numerical result.

The project may compare against:

- a clear PyTorch reference implementation;
- PyTorch scaled dot product attention;
- PyTorch FlexAttention where the workload matches.

The comparison must state which PyTorch path was used.

### Comparing FlashInfer

FlashInfer is designed for efficient attention and inference workloads, but not every FlashInfer operation is directly equivalent to every custom kernel in this project.

The project compares FlashInfer only when:

- the operation matches;
- the mask matches;
- the tensor layout matches or is converted fairly;
- the data type matches;
- the output definition matches;
- the benchmark includes the same work.

If a workload is not equivalent, the README records that limitation instead of presenting an invalid comparison.

---

## Profiling methodology

The project uses two types of Nsight analysis.

### Nsight Systems

Nsight Systems is used to understand the larger execution timeline:

- kernel launch sequence;
- CPU to GPU interaction;
- synchronization;
- gaps between kernels;
- repeated operations;
- overall execution flow.

### Nsight Compute

Nsight Compute is used to inspect individual kernels:

- achieved occupancy;
- memory throughput;
- global memory efficiency;
- shared memory behavior;
- register usage;
- instruction throughput;
- warp stalls;
- arithmetic intensity;
- compute and memory utilization.

The goal is to connect the profile to a design decision.

For example:

```text
Observed result:
The custom kernel reduces latency.

Possible explanation:
It avoids writing a full attention score matrix to global memory.

Evidence needed:
Lower global memory traffic and a profile showing the changed memory behavior.
```

The project will not describe a kernel as optimized merely because it is custom written. The performance claim must be supported by timing and profiling evidence.

---

## Triton and CUDA C++ roles

### Triton

Triton is used for rapid kernel development and controlled GPU experimentation.

It allows the project to explore:

- tiled computation;
- memory layouts;
- warp count;
- pipeline stages;
- masking;
- fused operations;
- autotuning parameters.

Triton is the first kernel backend because it allows a smaller implementation loop while still exposing important GPU decisions.

### CUDA C++

CUDA C++ is used for a lower level implementation of a simple supported workload.

The CUDA implementation is exposed to PyTorch through a C++ or CUDA extension. This demonstrates:

- C++ and CUDA integration;
- explicit thread and block organization;
- device memory reasoning;
- synchronization;
- compilation and extension loading;
- framework integration.

The project does not assume that CUDA C++ is always faster than Triton. The comparison is measured for a defined workload.

---

## Optional AI assisted optimization

The AI helper is deliberately not the foundation of the project.

It may propose:

- a tile size;
- a warp count;
- a pipeline configuration;
- a memory layout;
- a local Triton change;
- a local CUDA change;
- a loop or reduction transformation.

The helper cannot approve its own suggestion.

Every suggestion follows the same gate:

```text
propose a change
        |
        v
compile the candidate
        |
        v
run correctness tests
        |
        v
run the same benchmark
        |
        v
keep only if correct and faster
```

If time is limited, the project remains complete without the AI helper. The verified kernel and autotuning pipeline are the primary contribution.

---

## Planned build order

### Stage 1: Define the specification

Deliverables:

- specification format;
- parser;
- validation of required fields;
- clear representation of supported workloads.

### Stage 2: Implement the PyTorch reference

Deliverables:

- causal prefill reference;
- clear mask handling;
- stable softmax;
- defined tensor layouts;
- reference tests.

### Stage 3: Implement one restricted Triton kernel

Initial scope:

- causal prefill;
- one supported data type;
- one or two supported head dimensions;
- clearly defined tensor layout;
- no unnecessary generality.

Deliverables:

- Triton kernel;
- PyTorch comparison;
- first correctness test.

### Stage 4: Build the correctness harness

Deliverables:

- randomized tests;
- edge case tests;
- absolute and relative error checks;
- dtype specific tolerances;
- readable failure messages;
- saved test configuration.

### Stage 5: Build the benchmark harness

Deliverables:

- CUDA event timing;
- warm up handling;
- synchronization;
- repeated trials;
- median latency;
- environment metadata;
- separate compilation and steady state timing.

### Stage 6: Add autotuning

Deliverables:

- legal configuration list;
- correctness filter;
- timing loop;
- fastest correct configuration;
- saved tuning result.

### Stage 7: Add sliding window attention

Deliverables:

- reference implementation;
- window boundary tests;
- Triton implementation;
- benchmark comparison;
- explanation of skipped work and boundary handling.

### Stage 8: Add GQA decode

Deliverables:

- defined key value cache layout;
- query to key value head mapping;
- decode reference;
- decode kernel;
- cache and boundary tests;
- prefill versus decode benchmark separation.

### Stage 9: Add CUDA C++

Deliverables:

- simple CUDA C++ kernel;
- PyTorch extension interface;
- correctness comparison with Triton and PyTorch;
- compilation instructions;
- benchmark result for the same workload.

### Stage 10: Profile and document

Deliverables:

- Nsight Systems trace;
- Nsight Compute report;
- bottleneck analysis;
- explanation of memory, compute, occupancy, and launch behavior;
- reproducible result metadata.

### Stage 11: Add optional AI assistance

Deliverables:

- candidate proposal interface;
- isolated code change;
- automatic compilation;
- correctness gate;
- benchmark gate;
- accepted and rejected proposal log.

---

## What is intentionally out of scope

To keep the project technically complete and understandable, version 1 does not attempt to support everything.

Out of scope for the initial version:

- every possible attention mask;
- every GPU architecture;
- every model architecture;
- distributed multi GPU execution;
- production serving;
- arbitrary natural language to arbitrary CUDA code generation;
- universal optimality;
- formal proof of correctness for all possible inputs;
- unsupported FlashInfer comparisons;
- unmeasured performance claims.

The project focuses on producing a reliable and explainable result for a defined workload family.

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

## Requirements

The intended environment is:

- Linux;
- CUDA capable NVIDIA GPU;
- compatible NVIDIA driver;
- CUDA toolkit;
- Python;
- PyTorch;
- Triton;
- C++ compiler;
- FlashInfer for matching comparison workloads;
- Nsight Systems;
- Nsight Compute.

The exact versions must be recorded because kernel performance and package compatibility are environment dependent.

Each result should record:

```text
GPU:
Driver:
CUDA:
PyTorch:
Triton:
FlashInfer:
Python:
Operating system:
Commit:
```

---

## Reproducibility contract

A result is considered reproducible only when the repository provides:

1. The attention specification.
2. The input shapes.
3. The data type.
4. The mask definition.
5. The reference implementation.
6. The kernel implementation.
7. The tuning configuration.
8. The correctness tolerances.
9. The benchmark command.
10. The hardware and software metadata.
11. The raw benchmark output.
12. The profiling command when profiling is reported.

A single latency number without this information is not treated as a complete result.

---

## How this project matches the NVIDIA role

The target role emphasizes deep learning systems software, efficient inference and training, GPU kernels, code generators, extensible abstractions, runtimes, PyTorch, FlashInfer, CUDA C++, Triton, and performance optimization.

This project maps to those areas as follows:

| NVIDIA role area | Project component |
|---|---|
| Efficient inference and training technologies | Prefill attention and decode attention workloads |
| Designing and optimizing kernels | Triton and CUDA attention kernels |
| Code generators | Attention specification to implementation pipeline |
| Extensible abstractions | Structured attention specification and workload representation |
| Deep learning frameworks | PyTorch reference and PyTorch extension integration |
| FlashInfer ecosystem | Fair comparison for matching workloads |
| GPU architecture awareness | Tiling, memory traffic, warps, registers, occupancy, and cache behavior |
| Performance optimization | Autotuning and CUDA event benchmarking |
| Domain specific compiler ideas | Specification, validation, lowering, tuning, and execution stages |
| C/C++ skills | CUDA C++ kernel and PyTorch extension |
| Numerical reliability | Reference comparison, randomized tests, edge cases, and tolerances |
| Runtime and library engineering | Reusable benchmark, validation, and kernel selection pipeline |
| Technical collaboration and open source readiness | Clear interfaces, reproducible results, documented limitations, and modular repository structure |

The project is especially relevant because it does not treat a kernel as an isolated code sample. It treats the kernel as part of a larger library and runtime workflow:

```text
workload description
        +
reference behavior
        +
kernel implementation
        +
correctness gate
        +
autotuning
        +
runtime selection
        +
benchmark and profiling
```

---

## What an interviewer should understand after reading this README

After reading this README, the intended message is:

- The project is about GPU systems engineering, not only model training.
- Attention kernels are treated as both a correctness problem and a performance problem.
- The reference implementation is separated from the optimized implementation.
- Kernel configurations are selected by measurement, not intuition.
- Benchmarking is designed to avoid misleading comparisons.
- Triton and CUDA C++ are used for different levels of GPU control.
- Nsight is used to explain performance, not just produce screenshots.
- GQA decode is understood as a KV cache and memory access problem.
- The project has a clear scope and does not claim universal performance.
- Optional AI assistance is controlled by deterministic tests and measurements.

---

## Limitations

The first version has important limitations:

- Results are tied to the tested NVIDIA GPU.
- The best configuration may change for another GPU architecture.
- Numerical tolerances depend on data type and operation.
- FlashInfer comparisons are valid only for matching workloads.
- A passing test suite does not formally prove correctness for every possible input.
- A custom kernel is not automatically faster than a mature library.
- Autotuning can be expensive because it measures many configurations.
- GQA decode depends strongly on key value cache layout and access pattern.
- Benchmark results can change with driver, CUDA, PyTorch, Triton, and FlashInfer versions.

These limitations are part of the engineering result and will be reported rather than hidden.

---

## Interview explanation

A concise explanation of the project is:

> I am building a correctness-first autotuning pipeline for attention kernels. The user describes a supported attention workload using a small specification. The system creates a trusted PyTorch reference and a Triton or CUDA implementation. It validates the candidate across randomized shapes and edge cases using numerical error tolerances. Only correct candidates enter the autotuning search, where the system measures different tile, warp, and pipeline configurations using synchronized CUDA events. The fastest correct configuration is then compared with equivalent PyTorch and FlashInfer implementations, and Nsight is used to explain whether the bottleneck is memory movement, computation, occupancy, synchronization, or launch overhead. The optional AI component can propose changes, but automated correctness and performance gates decide whether those changes are accepted.

---

## References

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [Triton](https://github.com/triton-lang/triton)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [PyTorch FlexAttention](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)

