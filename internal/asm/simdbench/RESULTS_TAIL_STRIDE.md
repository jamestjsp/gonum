# AMD64 tail and stride follow-up

Measured on 2026-09-05 against fetched `codex/arm64-simd-blas` commit
`12e546269777a957f63c7ed7ea963e98e817998b`, using Go 1.27.1 on an Intel Core
i7-11850H. These changes improve experimental candidates; AMD64 production
assembly remains selected. No native ARM64 performance claim is made.

Six interleaved samples per case, prebuilt binaries, CPU2 affinity,
`GOMAXPROCS=1`, and the same benchmark harness on both revisions gave:

| Native width | Fetched SIMD wins vs ASM | New candidate wins vs ASM | New vs fetched SIMD: faster / slower / inconclusive |
|---|---:|---:|---:|
| 512 bits | 33/114 | 37/114 | 52 / 5 / 57 |
| 256 bits | 25/114 | 27/114 | 52 / 5 / 57 |

A win requires benchstat's per-case 5% significance test. Inconclusive does not
establish equivalence. Counts are unweighted and have no multiple-comparison
correction; they do not represent application throughput.

Selected 512-bit medians (ns/op, lower is better):

| Kernel | Fetched SIMD | New candidate | ASM |
|---|---:|---:|---:|
| f32 AXPY, n31 | 16.06 | 10.56 | 14.29 |
| c64 conjugated dot, n31 | 76.09 | 38.96 | 13.40 |
| c128 strided real scale, n4096 | 8052.50 | 2004.50 | 1561.00 |
| f32 strided widened dot, n4096 | 6667.00 | 2709.50 | 1960.50 |
| f64 cumulative sum, n4096 | 3827.00 | 2091.00 | 2284.50 |
| f64 cumulative product, n4096 | 3801.00 | 2128.00 | 2053.00 |

The cumulative-product comparison with ASM is inconclusive. Prefix improvements
come from fixed scalar blocks that preserve the original block/carry arithmetic
order and eliminate staging; they are not pure vector-arithmetic wins.

The [SIMD article supplied for this investigation](https://mitchellh.com/writing/everyone-should-know-simd)
frames load, compute, reduction and tail costs. Here the tuned ASM baseline
already uses SIMD where useful. Removing scratch packing, inlining a short
float32 tail leaf, narrowing complex64 short-dot dispatch at 256 bits, and using
exact-address sparse arithmetic provide more benefit than widening alone.

Extended coverage includes 448 boundary cases and 252 stride cases at each
width. At 512 bits, 128 boundary and 205 stride cases improved; at 256 bits,
110 boundary and 206 stride cases improved. All significant stride regressions
in this sweep use unit increments. The separate f64 matrix sweep exercises
unit-stride blocked paths and does not establish consumer gains from the new
sparse leaves. Every one of 21,336 timed measurements reported zero allocations.

Important retained costs and correctness repairs:

- Some short exact-size complex64 AXPY calls pay about 1–2 ns of helper setup;
  useful tails saved about 12–18 ns in the crossover review. Sparse ASM often
  retains a substantial lead through fewer bounds/address operations.
- The final manifest has five significant regressions per width. At 512 bits,
  these are f32 DotUnitary n4096 (+6.63%), Ger n8/n64 (+3.89%/+2.71%),
  f64 AxpyUnitary n31 (+3.23%), and L1NormInc n4096 (+7.11%). At 256 bits,
  they are f32 Ger n8 (+6.42%), f64 AddConst n31 (+2.52%), AxpyUnitary n31
  (+2.73%), L2DistanceUnitary n4096 (+2.34%), and ScalUnitaryTo n31 (+7.13%).
  L1NormInc differs between manifest and stride sweeps; unchanged routines also
  show layout-sensitive timing shifts. See the full report for the raw evidence.
- Go 1.27.1 native partial-load APIs can overread short slices. A guard-page
  regression reproduces a fault in the fetched f32 Ger implementation. Full
  bounded chunks plus scalar remainders replace those unsafe tails.
- Exceptional reductions retry the established accumulation order before any
  sequential recovery; regrouping can otherwise overflow a previously finite
  result. Compensated float64 norms are unchanged.
- New native helpers requiring AVX2 check it explicitly, including when portable
  SIMD selects 128 bits on an AVX-only CPU. CPU-feature-disabled tests verify
  fallback selection. Other architectures retain their established algorithms.
- Cumulative-product timing uses reciprocal factors with finite normal prefixes.
  The earlier input overflowed at element 1985; those timings were rejected.

Validation passed: full experimental test suite; numeric/BLAS tests at
emulated/128/256/512 widths; AVX2-disabled fallback checks; race, default, safe,
noasm and Go 1.24 default configurations; ARM64 cross-builds; numerical, overlap,
stride and guard-page regressions; formatting and repository policy checks.

The workstation used its powersave governor and did not isolate SMT sibling
CPU10. Main cases use 50 ms samples and boundary/stride cases use 20 ms; small
changes require caution. Raw logs, benchstat summaries, source/binary hashes,
rejected screens and the full report accompany the patch bundle. The incremental
patch targets 12e54626; a complete alternative targets e925697f and includes the
intervening fetched source plus earlier complex64 AXPY ASM fixes. Apply only the
patch matching the clean checkout's base.

## ARM64 integration check

Integrated on 2026-09-05 from the incremental 12e54626 patch. All 34 kernel/test
source hashes match the AMD64-tested manifest; no numerical implementation was
changed during import. The original CumProd timing caveat is also recorded in
the historical [BLAS report](RESULTS_BLAS.md).

A separate native Apple M1 Pro comparison used Go 1.27.1, `GOEXPERIMENT=simd`,
`GODEBUG=simd=128`, `GOMAXPROCS=1`, prebuilt binaries and six interleaved samples
against 12e54626. Both revisions used the same corrected benchmark harness.
Compilation and validation were stopped before timing. Public BLAS/SVD samples
used 50 ms; the 114 candidate cases used 30 ms.

None of the 22 selected public BLAS or four SVD cases changed significantly.
The candidate-only sweep had 3 faster, 8 slower and 103 inconclusive cases under
per-case 5% tests without multiple-comparison correction. The largest measured
candidate regression was c128 DotuInc n31: 56.58 to 60.89 ns (+7.60%, p=0.002).
Its n4096 case and the corresponding unitary cases also regressed about 4.6–5.7%.
These remain comparison candidates on ARM64, not new production dispatch.
No particular hardware cause was established for the timing shifts.

With corrected finite inputs, ARM64 CumProd n4096 measured 2.663 versus 2.720 µs
(p=0.855); this does not establish a difference. Candidate and public BLAS
allocations stayed zero. SVD allocation counts were unchanged.

Local validation passed the full Go 1.27.1 SIMD and Go 1.26.4 default suites;
affected emulation, safe and noasm tests including LAPACK/mat; numeric/BLAS race
and bounds checks; formatting, import policy, copyright and diff checks. Native
AMD64 timings and Linux guard-page execution are supplied-bundle evidence, not
new measurements on this ARM64 host. No new cross-compilation was performed.
