# AMD64 assembly versus Go SIMD

This package provides a Go 1.27 portable-SIMD candidate for every BLAS-related
AMD64 assembly entry point under `internal/asm`. It does not change production
dispatch.

On an AMD64 machine with Go 1.27.1 or newer, run the assembly-equivalence tests
and the same-binary comparison benchmarks with:

```sh
GOEXPERIMENT=simd go test ./internal/asm/simdbench
GOMAXPROCS=1 GOEXPERIMENT=simd go test ./internal/asm/simdbench \
  -run '^$' -bench '^BenchmarkAMD64AssemblyVsSIMD$' -benchmem -count=10 \
  | tee amd64-simd.txt
benchstat -col /implementation amd64-simd.txt
```

Install `benchstat` with
`go install golang.org/x/perf/cmd/benchstat@latest` if needed.

The checked manifest test fails when an assembly symbol is added or removed,
when its SIMD candidate is absent, when a candidate does not reach a `simd`
operation, or when its benchmark/equivalence runner is missing.
