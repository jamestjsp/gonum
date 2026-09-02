---
name: gonum-simd
description: Implement, review, or benchmark Go SIMD kernels in Gonum BLAS. Use for simd or archsimd adoption, assembly replacement, target-specific dispatch, and numerical performance validation; do not use for unrelated Gonum changes or generic Go optimization.
---

# Gonum SIMD

Improve Gonum SIMD code without changing behavior or slowing supported fallbacks.

## Refresh the experiment

Before editing, verify the current Go release, package documentation, build requirements, and proposal status. Experimental SIMD APIs are not covered by the Go 1 compatibility promise.

Use these primary sources:

- Go release notes: <https://go.dev/doc/devel/release>
- SIMD direction and architecture-specific API: <https://github.com/golang/go/issues/73787>
- Portable, vector-size-agnostic API: <https://github.com/golang/go/issues/78902>
- ARM64 SVE proposal: <https://github.com/golang/go/issues/79781>
- Default enablement discussion: <https://github.com/golang/go/issues/78979>
- CPU-feature checking proposal: <https://github.com/golang/go/issues/76175>

Do not assume that `GOEXPERIMENT=simd`, package paths, method names, build tags, vector widths, or CPU-feature obligations remain unchanged. Treat proposals as direction, not commitments.

## Find the real kernel

Trace the public BLAS operation into `internal/asm` and identify every important downstream caller before optimizing. Gonum's AMD64 assembly is useful algorithmic evidence, but ARM64 intrinsics need their own generated-code and benchmark validation.

Keep the SIMD implementation behind precise architecture, Go-version, experiment, and opt-out build constraints. Preserve scalar behavior for older Go releases, unsupported architectures, `safe`, `noasm`, and GCCGo unless current repository policy says otherwise.

Prefer the portable `simd` API when current toolchains make it competitive and it removes architecture-specific duplication. Retain `simd/archsimd` when a required operation or measured performance still demands it.

## Preserve semantics

Add persistent tests for:

- empty and short inputs, vector boundaries, and scalar tails;
- in-place operation and permitted overlap or aliasing;
- length and increment behavior already supported by the scalar path;
- numerical tolerances affected by reassociation or fused operations.

Partial overlap may require a scalar path even when exact in-place operation is safe. Do not infer numerical parity from compilation alone.

## Prove generated code and performance

Inspect the compiled kernel with `go tool objdump` or compiler diagnostics and confirm that the intended vector instructions are emitted without new hot-loop bounds checks or allocations.

Benchmark on the actual target CPU against a clean worktree at the current base commit, using the same toolchain, experiment flags, environment, and `GOMAXPROCS`. Use repeated interleaved samples and `benchstat`; stop and rerun when unrelated compilation or thermal/load interference contaminates the host.

Measure both focused kernels and representative BLAS consumers. Include small sizes to expose SIMD overhead, large sizes for throughput, transpose and shape variants where relevant, and allocation counts. Do not promote a cutoff or dispatch rule from one shape, worker count, or architecture.

## Validate the matrix

Run affected-package tests first, then the full suite with SIMD enabled. Also test the default supported Go version and the `safe` and `noasm` paths. Cross-build relevant ARM64 and AMD64 targets, and run race, formatting, import-policy, copyright, and diff checks required by the repository.

Report the exact base commit, CPU, OS, Go version, experiment flags, sample count, statistical result, allocations, branch, and commit/push state. Separate measured facts from roadmap-driven follow-up work.
