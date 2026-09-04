# Portable Go SIMD candidates

Portable SIMD candidates live beside the routines they may replace in
`internal/asm/{f32,f64,c64,c128}/simd.go`. They are architecture-neutral and
cover every BLAS-related AMD64 assembly entry point. This package keeps only the
coverage manifest, equivalence tests, and comparison benchmarks.

The candidates do not change production dispatch. Current AMD64 assembly and
the tuned ARM64 implementations remain active until a candidate is faster on
the target architecture. On AMD64, `current` below means assembly; on ARM64 it
means the current ARM64 implementation.

With Go 1.27.1 or newer, run the same-binary equivalence tests and benchmarks
with:

```sh
GOEXPERIMENT=simd go test ./internal/asm/simdbench
GOMAXPROCS=1 GOEXPERIMENT=simd go test ./internal/asm/simdbench \
  -run '^$' -bench '^BenchmarkCurrentVsSIMD$' -benchmem -count=10 \
  | tee simd.txt
benchstat -col /implementation simd.txt
```

Install `benchstat` with
`go install golang.org/x/perf/cmd/benchstat@latest` if needed.

The checked manifest fails when an AMD64 assembly symbol is added or removed,
when its package-local candidate is absent, when a candidate does not reach a
`simd` operation, or when its benchmark/equivalence runner is missing.

When portable SIMD becomes stable, recheck the package path, build constraint,
vector-width contract, generated code, and benchmark crossovers before changing
dispatch. Removing `goexperiment.simd` is intentionally a small boundary
change; it is not assumed to be the only migration Go will require.
