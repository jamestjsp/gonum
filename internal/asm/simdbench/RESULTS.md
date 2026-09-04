# SIMD optimization follow-up, 2026-09-04

This measures the improved portable candidates against the previous portable
candidates. It does **not** measure improvements over AMD64 assembly or an
end-to-end BLAS/SVD speedup. Production dispatch is unchanged.

## Reproduction

- Host: Apple M1 Pro, macOS 26.6.2 (25G83), darwin/arm64.
- Go 1.27.1, GOEXPERIMENT=simd, GOMAXPROCS=1; 128-bit vectors.
- Baseline kernels: 2ed36cad, with the corrected comparison harness committed
  as f900637c3357e28324fc8b2cf5e46752e37b1b96.
- Candidate: the source changes accompanying this report.
- Separate binaries built before timing. Six samples per case at 50ms;
  baseline/candidate run order reversed on alternating samples.
- Both binaries use identical stable benchmark inputs. Analysis uses benchstat.
  The full comparison covers 57 symbols at two sizes. No timed compilation.

Build each checkout with:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -o simdbench.test ./internal/asm/simdbench
```

For each sample, run the old and new binaries in alternating order with:

```sh
GOMAXPROCS=1 ./simdbench.test -test.run '^$' \
  -test.bench 'BenchmarkCurrentVsSIMD/.*/.*/.*/implementation=simd$' \
  -test.benchmem -test.benchtime=50ms -test.count=1
```

## Selected measured results

All rows below have p=0.002, n=6, and zero allocations.

| Candidate | Size | Previous | Improved | Time change |
| --- | ---: | ---: | ---: | ---: |
| f32 Ger | 64×64 | 6.089 µs | 1.221 µs | -79.95% |
| f64 GemvN | 64×64 | 5.396 µs | 1.578 µs | -70.77% |
| f64 GemvT | 64×64 | 9.386 µs | 2.058 µs | -78.07% |
| f64 Ger | 64×64 | 9.202 µs | 2.057 µs | -77.65% |
| f64 DotUnitary | 4096 | 2.408 µs | 1.103 µs | -54.19% |
| f64 Sum | 4096 | 1.893 µs | 565.9 ns | -70.10% |
| f64 CumSum | 4096 | 9.502 µs | 3.109 µs | -67.28% |
| c128 AxpyInc | 4096, inc=2 | 16.000 µs | 9.555 µs | -40.28% |

Short cases also improved: f64 DotUnitary at n=31 changed from 20.81ns to
15.11ns (-27.41%); CumSum changed from 46.45ns to 26.69ns (-42.54%).

The full screening run caught a 6–7% ARM64 regression in large complex unitary
dots after integer staging. Exact-width input slices removed it. An independent
eight-sample, 100ms alternating rerun of the final complex dot loops measured:

| Candidate, n=4096 | Previous | Improved | Time change |
| --- | ---: | ---: | ---: |
| c128 DotuUnitary | 9.560 µs | 8.911 µs | -6.79%, p<0.001 |
| c128 DotcUnitary | 8.926 µs | 8.909 µs | -0.18%, p=0.004 |
| c64 DotuUnitary | 5.746 µs | 5.735 µs | -0.19%, p=0.004 |
| c64 DotcUnitary | 5.750 µs | 5.738 µs | -0.22%, p=0.010 |

Treat changes below 1% as practically unchanged. Final complex dot short cases
improved 7–17%. L2 norm edits were discarded after short-case regressions of
about 5–6%; their previous implementations remain. Some general-stride and
mixed-precision cases remain substantially limited by staging.

## AMD64 evidence and next measurement

Windows/AMD64 Go 1.27.1 compiler listings verify that the selected AVX256 real
strided loops use integer MOVL/MOVQ lane transfers in place of legacy
MOVSS/MOVSD. Complex AXPY alpha vectors are constructed before the loop and the
multiply-add helper inlines. These changes remove the inspected scalar SSE
moves from those vector loops. Scalar tails, reductions, mixed-precision
conversion, and prefix arithmetic still need further work.

The f64 Sum AVX256 clone has a 40-byte nosplit frame instead of the previous
264-byte frame. Exact-width scratch is compiler-specialized stack storage;
persistent allocation checks cover every candidate.

The magnitude of any AMD64 benefit is unmeasured here. Run the corrected
[Windows comparison](README.md) on the issue's i7-1270P with the original
affinity and flags, then compare current assembly with the new candidates.
Keep assembly as the production path until native evidence supports a change.
