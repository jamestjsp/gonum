# Portable Go SIMD candidates

SIMD candidates live beside the routines they may replace in
`internal/asm/{f32,f64,c64,c128}/simd.go` and cover every BLAS-related AMD64
assembly entry point. Portable operations are shared across architectures.
Small AMD64 leaves handle complex permutations and widened dot products where
the portable API or compiler-generated conversions would require scalar staging.
Other targets retain portable fallbacks. This package keeps only the
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

The `go` directive in `go.mod` does not need to change: the Go 1.27 toolchain
and `GOEXPERIMENT=simd` satisfy the source build constraints.

On Windows PowerShell, the complete comparison can be run with:

```powershell
$env:GOTOOLCHAIN = "go1.27.1"
$env:GOEXPERIMENT = "simd"
$env:GOAMD64 = "v1"
$env:GOMAXPROCS = "1"
go test ./internal/asm/simdbench
go test ./internal/asm/simdbench -run '^$' `
  -bench '^BenchmarkCurrentVsSIMD$' -benchmem -benchtime=200ms -count=10 -timeout=0 `
  | Tee-Object -FilePath amd64-simd.txt
benchstat -col /implementation amd64-simd.txt
```

Record `git rev-parse HEAD`, `go version`, CPU model, OS, selected vector width,
and processor affinity with the results. For a hybrid Intel CPU, use the same
P-core affinity for each run. `GOAMD64=v1` still permits portable SIMD to select
supported wider vectors at runtime.

## Issue 6 follow-up

See [measured ARM64 results and AMD64 instruction findings](RESULTS.md) for
the optimization follow-up. See [native AMD64 tuning results](RESULTS_AMD64.md) for the subsequent
comparison against assembly on an AVX512 host.

The [initial AMD64 results](https://github.com/jamestjsp/gonum/issues/6#issuecomment-5541048813)
identified oversized scratch buffers, unnecessary staging for contiguous
matrix rows, and decaying benchmark inputs. Most scratch now uses `make([]T, width)`:
Go 1.27 specializes the width before escape analysis, producing stack storage
for the selected vector size. This also avoids imposing a fixed future vector
width ceiling. The allocation regression test must stay green on each target.
The robust L2 norm kernels retain their prior implementation: this scratch
change regressed short calls on ARM64 and did not improve long calls.

Real contiguous increment operations use their unitary candidates. Dot and
sum candidates use four independent accumulators; explicit load spans reduce
redundant bounds checks. General strides, mixed-precision conversion, complex
deinterleaving, and prefix scans still need staging with the current portable
API. These remain candidates for further algorithm or compiler improvements.

AMD64 code inspection also found legacy SSE scalar moves alternating with AVX
vector operations inside staging loops. Integer memory views now move lane
bits into and out of unsigned scratch, then `BitsToFloat32`/`BitsToFloat64`
reinterpret them for vector arithmetic. Unlike scalar `math.Float64bits`
calls, these views preserve integer moves through Go 1.27 optimization.
Complex alpha broadcasts are hoisted outside the vector loop as well.
[Intel documents penalties for AVX/SSE mixing](https://www.intel.com/content/dam/develop/external/us/en/documents/11mc12-avoiding-2bavx-sse-2btransition-2bpenalties-2brh-2bfinal-809104.pdf).
Removing these instructions is verified in Windows/AMD64 compiler output;
their contribution to the office timings still needs native measurement.

Mixed-precision `Ddot`, prefix scans, and robust norm recurrences still mix
scalar arithmetic with vectors. Efficient widening/reduction/scan operations
or compiler improvements are the next opportunities. Recheck generated
instructions with future Go releases before retaining a source workaround.

In-place scaling and division benchmarks use unit-magnitude factors to avoid
subnormal decay. Numerical equivalence tests retain the original factors.
Old timings for the seven affected in-place `Div`, `Scal`, and `Dscal` cases
cannot be compared directly with corrected timings. When comparing source
revisions, apply the corrected benchmark harness to both checkouts.

The checked manifest fails when an AMD64 assembly symbol is added or removed,
when its package-local candidate is absent, when a candidate does not reach a
`simd` operation, or when its benchmark/equivalence runner is missing.
Equivalence tests cover empty inputs and vector boundaries, and allocation
tests cover all 57 current and candidate entry points.

When portable SIMD becomes stable, recheck the package path, build constraint,
vector-width contract, generated code, and benchmark crossovers before changing
dispatch. Removing `goexperiment.simd` is intentionally a small boundary
change; it is not assumed to be the only migration Go will require.

The native AMD64 follow-up keeps production dispatch unchanged. Contiguous
complex kernels use interleaved vectors, matrix kernels share loads across rows,
and ordinary L2 norms use vector sums of squares with a scaled fallback for
extreme magnitudes. The widening and complex-shuffle AMD64 leaves avoid Go
1.27.1 `FromArch` stack copies in hot loops. Recheck these workarounds when the
compiler or portable API changes. Small calls, arbitrary strides, and matrix
shapes still have different crossovers; native timing is required before changing
production dispatch.
