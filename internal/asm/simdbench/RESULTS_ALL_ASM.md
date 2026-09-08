# Gonum SIMD versus AMD64 assembly — measured checkpoint

Detailed raw evidence accompanies the Samsung patch bundle. Evidence paths below refer to its `evidence/` directory; the binary copies are in `measured-binaries/`.

This patch improves the Go SIMD candidates on the native Intel i7-11850H, but does not beat assembly in every routine or shape. The overall optimization goal remains open. No unmeasured case is classified as theoretically impossible.

Base: `93a1976af7d80c677589f2e9d1f76ad2ed152d87` on `origin/codex/arm64-simd-blas`. The prior SIMD comparator is local commit `81b40a257c34b14b82646ca7163271d800b7240f`, which includes the earlier accepted tuning. The bundle SOURCE.json pins the delivered commit and verified tree. The final measured source hashes are in `selected-v2-final-source-sha256.json`; raw samples and binary hashes are in `selected-v2-acceptance/`.

Native measurements use Linux AMD64, Go 1.27.1, `GOEXPERIMENT=simd`, `GOAMD64=v1`, `GOMAXPROCS=1`, CPU 2, and separately selected 512-bit and 256-bit SIMD widths. ARM64 was cross-built only; the branch name is not ARM64 performance evidence. Existing handwritten assembly already uses SIMD instructions.

| Comparison | Cases | Prior SIMD ASM wins | Selected ASM wins, median | Significant ASM wins | Significant regressions of at least 5% versus prior SIMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| manifest-512 | 114 | 48 | 87 | 85 | 1 |
| manifest-256 | 114 | 46 | 86 | 83 | 0 |
| consumers-512 | 402 | 59 | 209 | 205 | 6 |
| consumers-256 | 402 | 43 | 209 | 205 | 1 |

The kernel manifest covers all 57 BLAS assembly entry points at two sizes per width. 33 routines win all four tested size/width medians; 31 win all four significantly. The complete per-routine result, including every remaining loss, is in `selected-v2-acceptance/routine-coverage.csv` and JSON. These counts are coverage summaries, not a workload-weighted speedup estimate.

The 402 consumer cases execute real BLAS methods with matched temporary method-local overlays. They include triangular products, norms, rank-one updates, transposed and nontransposed matrix-vector products, complex dots, widened float32 dots, and complex scaling. Every variant receives identical finite inputs, mutation resets, and fixture checks. The 68 widened-dot, 72 complex-scale, 34 complex cutoff, 32 contiguous GER, and 34 small-GemvT cases deliberately cover important boundaries; their frequency is not an application workload model.

AMD64 production BLAS dispatch remains on its existing implementations. The SIMD candidates are directly callable for measurement, and real consumer candidate routing is reproduced by `build-consumers-v2.py`. For positive-stride `Zdscal`, unchanged production uses scalar Go; the comparator labeled ASM explicitly routes that loop to `DscalInc` assembly. Separate `zdscal-production-*` timings retain the actual scalar production comparison. The Go 1.27 AMD64 `internal/math32.Sqrt` intrinsic change does affect production, and is measured separately against the prior assembly with `sqrt-production-*` samples.

The selected changes include direct short reduction entries, bounded native tails and strides, exact gathered GER tiles with reuse across rows, a wide contiguous GER path, native small GEMV paths, complex short-dot and scaling improvements, and the corrected explicit AVX2 admission for widened dots. The 512-bit c64 dot dispatch now uses its faster native256 path through 511 elements. At width256, ScalUnitaryTo uses the measured native256 loop at long lengths as well. The final report counts come from the integrated binary, not a sum of isolated prototype wins.

Several tempting variants were rejected: seeded short float32 reductions, packed short float64 Sum, a larger GER tile that spilled registers, a ScalUnitaryTo unroll rollback, and complex FMA changes that altered exceptional-value behavior. The corresponding source, disassembly, raw samples and failure explanations remain in the evidence. Wider vectors, fewer apparent operations, or more unrolling did not by themselves establish an improvement.

The remaining gaps are concentrated in short reductions, some short strided operations, and awkward matrix shapes. Source and generated-code work has exposed repeated eligibility checks, stack and register moves, lane extraction and packing, bounds/address work, and shape branches. Existing assembly also pipelines independent row products effectively. These are observed mechanisms and follow-up hypotheses, not proof that every loss has one cause or that further improvements are impossible. The AddConst entry and both native256/native512 lowerings are instruction-identical after relocation normalization in the final and prior binaries. Its measured regression is therefore a placement or surrounding-state effect, not a changed arithmetic body; the exact cause is not yet isolated. Cumulative source bodies also showed placement-sensitive shifts in earlier passes. Cached startup eligibility and small-GemvT row pipelining remain separate, unaccepted follow-ups.

Each main comparison uses six alternating samples with 50 ms timed work per case; the scalar sqrt comparison uses 150 ms. `simdbenchclean` clears upper AVX state before both implementations. The authorized background menu helpers are paused and restored by the quiet runner, and no compilation, tests or profiling run during timing. `benchstat` output and exact two-sided rank-test probabilities are retained. Significance is per-case and unadjusted for multiple comparisons; marginal differences are not universal claims. All 19,320 recorded observations report zero bytes and zero allocations per operation; the raw records and allocation-check.json retain that proof.

The integrated source passed 17 validation stages: kernel suites at 0/128/256/512, independently disabled AVX2 configurations, FMA-disabled coverage, race, checkptr correctness, the full repository SIMD suite, default/safe/noasm paths, Go 1.24 kernel and BLAS compatibility, ARM64 full-repository compilation, and diff checks. Additional full BLAS candidate-overlay suites passed at 512 and 256; all 402 named consumer fixtures were counted and passed in each of eight width/feature configurations. Formatting, import policy, and copyright checks passed. Numerical checks retain cancellation, overflow/NaN classification, alias, exact-tail and protected-gap behavior without relaxed tolerances. Normal allocation checks remain enabled. Only checkptr2 excludes allocation assertions because that instrumentation deliberately moves certain unsafe conversions to the heap; its correctness checks still run.

Feature overrides on this capable processor validate routing, not execution on physically AVX-only hardware. Generated-code admission inspection is therefore also required and retained. Forcing width512 after disabling the AVX512 hardware bundle correctly panics during Go SIMD initialization; the supported width256 configuration tests that fallback. An earlier Dger test-selection mistake was corrected with exact test names and child-case counts. The final benchmark preflight similarly caught and corrected an overestimated count in the separate Zdscal production selection. Initial failed attempts are retained with their explanations.

The export script freshly fetches origin, requires the exact base, replays the patch series and combined diff in separate clean worktrees to identical trees, and verifies preservation of all 56 original dirty source files. It writes a new Samsung bundle and rereads every copied file against SHA256SUMS. The previous USB bundle is retained. The prior c64 assembly alignment and empty-tail repair `d224925a` is already in the base; its reference patch is included under `already-in-upstream` and must not be applied again. No remote push is performed.

Reproduce the kernel comparison after applying the patch with:

```sh
GOEXPERIMENT=simd GOMAXPROCS=1 GODEBUG=simd=512 go test -tags simdbenchclean ./internal/asm/simdbench -run '^$' -bench '^BenchmarkCurrentVsSIMD$' -benchtime=100ms -count=6
```

Repeat at `simd=256` on a quiet, pinned native CPU. For matched real consumers, regenerate the overlays with `build-consumers-v2.py --candidate-root <patched-checkout> --base-root <prior-81b40a25-checkout> --output <overlay-directory>`, then compile each corresponding variant using the commands in `final-build/status.json`, adapted to local paths. Exact measured Linux AMD64 binaries are also included in the bundle for reproducing the recorded executable rather than relying on a newer toolchain.

Measured significant regressions of at least 5% against prior SIMD are retained below. A negative ASM delta still means faster than ASM.

| Suite and case | Prior ns | Selected ns | ASM ns | Change vs prior | Change vs ASM |
| --- | ---: | ---: | ---: | ---: | ---: |
| consumers-256: `BenchmarkSIMDComplexScaleConsumers/Zdscal/n=1/inc=1` | 7.379 | 8.011 | 4.999 | +8.6% | +60.3% |
| consumers-512: `BenchmarkSIMDComplexScaleConsumers/Zdscal/n=1/inc=1` | 7.702 | 8.692 | 4.962 | +12.9% | +75.2% |
| consumers-512: `BenchmarkSIMDGerShapes/m=16/n=32/inc=1` | 65.31 | 70.25 | 71.97 | +7.6% | -2.4% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Dsdot/n=32/inc=1` | 11.57 | 12.55 | 10.88 | +8.5% | +15.4% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Sdsdot/n=32/inc=1` | 12.04 | 13.66 | 11.13 | +13.5% | +22.6% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Dsdot/n=64/inc=1` | 14.62 | 16.09 | 18.24 | +10.1% | -11.8% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Sdsdot/n=64/inc=1` | 15.48 | 17.07 | 19.3 | +10.2% | -11.6% |
| manifest-512: `BenchmarkCurrentVsSIMD/f64/AddConst/n=31/implementation=kernel` | 9.073 | 9.739 | 17.27 | +7.3% | -43.6% |
| scal-width-512: `BenchmarkSIMDBoundaries/f64/ScalUnitaryTo/n=64/stride=1/implementation=kernel` | 10.5 | 11.25 | 10.82 | +7.2% | +3.9% |

The separate production sqrt comparison at width512 measures 1.208 ns for the Go intrinsic versus 2.176 ns for the former ASM route (-44.5% median time).
The separate production sqrt comparison at width256 measures 1.2 ns for the Go intrinsic versus 2.178 ns for the former ASM route (-44.9% median time).
