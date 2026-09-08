// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"math/big"
	"os"
	"simd"
	"simd/archsimd"
	"strconv"
	"testing"
)

// This fixture is installed with exactly one separately bound chain-count file.
// The integer oracle does not reproduce either summation algorithm.
func l1MultiUnits(x float64) *big.Int {
	b := math.Float64bits(x) & 0x7fffffffffffffff
	e, f := (b>>52)&2047, b&((uint64(1)<<52)-1)
	if e == 2047 {
		panic("nonfinite integer-oracle input")
	}
	if e == 0 {
		return new(big.Int).SetUint64(f)
	}
	return new(big.Int).Lsh(new(big.Int).SetUint64(f|(uint64(1)<<52)), uint(e-1))
}

func l1MultiLengths() []int {
	n := []int{0, 1, 2, 7, 8, 9, 31, 32, 33, 63, 64, 65, 95, 96, 97, 126, 127}
	for i := 128; i <= 161; i++ {
		n = append(n, i)
	}
	for i := 255; i <= 289; i++ {
		n = append(n, i)
	}
	return append(n, 4095, 4096, 4097)
}

var l1MultiKinds = []string{
	"dyadic", "signed-zero", "minsub", "normal-boundary", "wide-exponents",
	"half-below", "half-exact", "half-above", "near-half-dense",
	"max-single", "max-pair", "positive-inf", "negative-inf", "nan", "inf-nan",
}

func l1MultiInput(n int, kind string) ([]float64, bool, bool) {
	x := make([]float64, n)
	// exact means each returning positive graph is exact for this fixture.
	exact, retry := false, false
	for i := range x {
		switch kind {
		case "dyadic":
			x[i] = float64(i%17-8) / 4
			exact = true
		case "signed-zero":
			x[i] = math.Copysign(0, float64(i%2*2-1))
			exact = true
		case "minsub":
			x[i] = math.Float64frombits(uint64(i%7 + 1))
			if i%2 != 0 {
				x[i] = -x[i]
			}
			exact = true
		case "normal-boundary":
			x[i] = math.Float64frombits((uint64(1) << 52) + uint64(i%3) - 1)
		case "wide-exponents":
			x[i] = math.Ldexp(1+float64(i%13)*0x1p-10, (i*73)%1901-1000)
			if i%2 != 0 {
				x[i] = -x[i]
			}
		case "near-half-dense":
			x[i] = math.MaxFloat64 / (2 * float64(n))
		}
	}
	if n == 0 {
		return x, true, false
	}
	switch kind {
	case "half-below":
		x[n-1] = math.Nextafter(math.MaxFloat64/2, 0)
		exact = true
	case "half-exact":
		x[n-1] = math.MaxFloat64 / 2
		exact = true
	case "half-above":
		x[n-1] = math.Nextafter(math.MaxFloat64/2, math.Inf(1))
		exact = true
		retry = true
	case "max-single":
		x[n-1] = math.MaxFloat64
		exact = true
		retry = true
	case "max-pair":
		x[0] = math.MaxFloat64
		x[n-1] = math.MaxFloat64
		retry = true
	case "positive-inf":
		x[n/2] = math.Inf(1)
		retry = true
	case "negative-inf":
		x[n/2] = math.Inf(-1)
		retry = true
	case "nan":
		x[n/2] = math.Float64frombits(0x7ff8000000000042)
		retry = true
	case "inf-nan":
		x[0] = math.Inf(-1)
		x[n-1] = math.Float64frombits(0x7ff8000000000042)
		retry = true
	}
	return x, exact, retry
}

func l1MultiSame(t *testing.T, got, want float64) {
	t.Helper()
	// Existing public semantics require NaN classification, not a NaN payload.
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("got %g, want NaN", got)
		}
		return
	}
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("bits got %016x want %016x", math.Float64bits(got), math.Float64bits(want))
	}
}

func l1MultiCheck(t *testing.T, x []float64, got float64, exact, retry bool) {
	t.Helper()
	w := simd.BroadcastFloat64s(0).Len()
	active := len(x) >= 128 && len(x) <= 1<<30 && !simd.Emulated() && archsimd.X86.AVX2() && w >= 1 && w <= 8
	old := l1MultiOldEntryOracle(x)
	if !active || retry {
		l1MultiSame(t, got, old)
	}
	A := new(big.Int)
	hasInf, hasNaN := false, false
	for _, v := range x {
		if math.IsNaN(v) {
			hasNaN = true
			continue
		}
		if math.IsInf(v, 0) {
			hasInf = true
			continue
		}
		A.Add(A, l1MultiUnits(v))
	}
	if hasNaN {
		if !math.IsNaN(got) {
			t.Fatalf("lost NaN: %g", got)
		}
		return
	}
	if hasInf {
		if !math.IsInf(got, 1) {
			t.Fatalf("absolute infinity: %g", got)
		}
		return
	}
	if math.IsInf(got, 1) {
		// Overflow fixtures must match the unchanged portable retry exactly.
		l1MultiSame(t, got, old)
		// A<=MaxFloat64 alone does not prove that the old rounded graph
		// remains finite near overflow; preserve its established result.
		if A.Cmp(l1MultiUnits(math.MaxFloat64/2)) <= 0 {
			t.Fatal("safe exact sum overflowed")
		}
		return
	}
	if math.IsNaN(got) || got < 0 {
		t.Fatalf("positive finite input returned %g", got)
	}
	D := l1MultiUnits(got)
	if A.Sign() == 0 {
		if math.Float64bits(got) != 0 {
			t.Fatal("zero sum is not +0")
		}
		return
	}
	if exact {
		if D.Cmp(A) != 0 {
			t.Fatal("exact dyadic/sparse sum changed")
		}
		return
	}
	// Short public entry is unchanged, checked bitwise above. Do not pretend
	// that it uses the portable long graph for a gamma-depth bound.
	if len(x) < 128 {
		return
	}
	oldDepth := len(x)/w + w + len(x)%w
	depth := oldDepth
	if active {
		k := l1MultiChains
		candidateDepth := len(x)/(k*w) + (k - 1) + w + len(x)%w
		if len(x)%(k*w) >= w {
			candidateDepth++
		}
		// gamma_d=d/(2^53-d). Prove (1+gamma_d)*A <= halfMax
		// using integers before selecting the shorter returning-graph bound.
		scale := new(big.Int).Lsh(big.NewInt(1), 53)
		lhs := new(big.Int).Mul(A, scale)
		rhs := new(big.Int).Mul(l1MultiUnits(math.MaxFloat64/2), new(big.Int).Sub(scale, big.NewInt(int64(candidateDepth))))
		if lhs.Cmp(rhs) <= 0 {
			depth = candidateDepth
		} else if candidateDepth > depth {
			depth = candidateDepth
		}
	}
	difference := new(big.Int).Sub(D, A)
	difference.Abs(difference)
	factor := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 53), big.NewInt(int64(depth)))
	lhs := new(big.Int).Mul(difference, factor)
	rhs := new(big.Int).Mul(A, big.NewInt(int64(depth)))
	if lhs.Cmp(rhs) > 0 {
		t.Fatalf("exact integer gamma bound failed n=%d width=%d chains=%d depth=%d", len(x), w, l1MultiChains, depth)
	}
}

func TestL1MultiOracleUnits(t *testing.T) {
	cases := []struct {
		x    float64
		want *big.Int
	}{
		{0, big.NewInt(0)}, {math.Copysign(0, -1), big.NewInt(0)},
		{math.SmallestNonzeroFloat64, big.NewInt(1)},
		{-math.SmallestNonzeroFloat64, big.NewInt(1)},
		{math.Float64frombits((uint64(1) << 52) - 1), new(big.Int).SetUint64((uint64(1) << 52) - 1)},
		{math.Float64frombits(uint64(1) << 52), new(big.Int).Lsh(big.NewInt(1), 52)},
		{1, new(big.Int).Lsh(big.NewInt(1), 1074)},
		{math.MaxFloat64, new(big.Int).Lsh(new(big.Int).SetUint64((uint64(1)<<53)-1), 2045)},
	}
	for _, c := range cases {
		if l1MultiUnits(c.x).Cmp(c.want) != 0 {
			t.Fatalf("units: %g", c.x)
		}
	}
}

func TestL1MultiRuntime(t *testing.T) {
	w := simd.BroadcastFloat64s(0).Len()
	t.Logf("L1_MULTI_RUNTIME chains=%d width=%d emulated=%t avx2=%t", l1MultiChains, w, simd.Emulated(), archsimd.X86.AVX2())
	for _, v := range []struct{ name, got string }{
		{"L1_MULTI_EXPECT_WIDTH", strconv.Itoa(w)},
		{"L1_MULTI_EXPECT_EMULATED", strconv.FormatBool(simd.Emulated())},
		{"L1_MULTI_EXPECT_AVX2", strconv.FormatBool(archsimd.X86.AVX2())},
	} {
		// Controlled runs may assert their startup mode; ordinary package
		// tests also run without experiment-specific environment variables.
		want, ok := os.LookupEnv(v.name)
		if ok && want != v.got {
			t.Fatalf("%s got %s want %s", v.name, v.got, want)
		}
	}
	if l1MultiChains != 2 && l1MultiChains != 4 {
		t.Fatal("invalid separately bound chain count")
	}
}

func TestL1MultiNumerical(t *testing.T) {
	for _, n := range l1MultiLengths() {
		for _, kind := range l1MultiKinds {
			t.Run(fmt.Sprintf("n%d/%s", n, kind), func(t *testing.T) {
				x, exact, retry := l1MultiInput(n, kind)
				saved := make([]uint64, n)
				for i, v := range x {
					saved[i] = math.Float64bits(v)
				}
				l1MultiCheck(t, x, L1NormSIMD(x), exact, retry)
				// Inc1 must restrict its view to n, including n=0; suffix NaNs
				// would expose reading the unused suffix. This does not invent
				// negative-stride or invalid-span behavior for the unchanged API.
				padded := append(append([]float64(nil), x...), math.NaN(), math.Inf(1))
				l1MultiCheck(t, x, L1NormIncSIMD(padded, n, 1), exact, retry)
				for i, v := range x {
					if math.Float64bits(v) != saved[i] {
						t.Fatal("direct input modified")
					}
				}
				for i, v := range padded[:n] {
					if math.Float64bits(v) != saved[i] {
						t.Fatal("Inc1 input modified")
					}
				}
				if !math.IsNaN(padded[n]) || !math.IsInf(padded[n+1], 1) {
					t.Fatal("Inc1 suffix modified")
				}
			})
		}
	}
}

func TestL1MultiInc1Contract(t *testing.T) {
	for _, n := range []int{0, -1} {
		l1MultiSame(t, L1NormIncSIMD(nil, n, 1), 0)
	}
	for _, n := range []int{1, 127, 128, 129} {
		t.Run(fmt.Sprintf("short-slice-n%d", n), func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("missing existing x[:n] panic")
				}
			}()
			L1NormIncSIMD(make([]float64, n-1), n, 1)
		})
	}
}

var l1MultiSink float64

func TestL1MultiZeroAllocations(t *testing.T) {
	for _, n := range []int{127, 128, 129, 159, 160, 161, 4096} {
		for _, kind := range []string{"dyadic", "half-above", "nan"} {
			x, _, _ := l1MultiInput(n, kind)
			for _, route := range []string{"direct", "inc1"} {
				t.Run(fmt.Sprintf("n%d/%s/%s", n, kind, route), func(t *testing.T) {
					run := func() {
						if route == "direct" {
							l1MultiSink = L1NormSIMD(x)
						} else {
							l1MultiSink = L1NormIncSIMD(x, n, 1)
						}
					}
					a := testing.AllocsPerRun(1024, run)
					t.Logf("L1_MULTI_ALLOC n=%d kind=%s route=%s calls=1024 allocs=%g", n, kind, route, a)
					if a != 0 {
						t.Fatalf("allocations: %g", a)
					}
				})
			}
		}
	}
}

// Each index is the sole contributing input in turn. Exact marker sums detect
// missing/duplicated lanes independently of the gamma accuracy allowance.
func TestL1MultiEveryInput(t *testing.T) {
	for _, n := range []int{128, 129, 159, 160, 161, 257} {
		for i := 0; i < n; i++ {
			t.Run(fmt.Sprintf("n%d/i%d", n, i), func(t *testing.T) {
				for _, v := range []float64{-2, -math.SmallestNonzeroFloat64} {
					x := make([]float64, n)
					x[i] = v
					l1MultiSame(t, L1NormSIMD(x), math.Abs(v))
					l1MultiSame(t, L1NormIncSIMD(x, n, 1), math.Abs(v))
				}
			})
		}
	}
}

func TestL1MultiSpecialPositions(t *testing.T) {
	for _, n := range []int{128, 129, 160, 161, 4096} {
		for _, i := range []int{0, n / 2, n - 1} {
			for j, v := range []float64{math.Nextafter(math.MaxFloat64/2, math.Inf(1)), math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()} {
				t.Run(fmt.Sprintf("n%d/i%d/value%d", n, i, j), func(t *testing.T) {
					x := make([]float64, n)
					x[i] = v
					l1MultiCheck(t, x, L1NormSIMD(x), true, true)
					l1MultiCheck(t, x, L1NormIncSIMD(x, n, 1), true, true)
				})
			}
		}
	}
}

// Allocation-only N=1 admission fixture. No timing or speed inference is made
// from these rows; numerical checks above are separate and remain mandatory.
func BenchmarkL1MultiAllocation(b *testing.B) {
	for _, n := range []int{127, 128, 129, 159, 160, 161, 4096} {
		for _, kind := range []string{"dyadic", "half-above", "nan"} {
			for _, route := range []string{"direct", "inc1"} {
				b.Run(fmt.Sprintf("n%d/%s/%s", n, kind, route), func(b *testing.B) {
					x, _, _ := l1MultiInput(n, kind)
					b.ReportAllocs()
					if archsimd.X86.AVX() {
						archsimd.ClearAVXUpperBits()
					}
					b.ResetTimer()
					if route == "direct" {
						for i := 0; i < b.N; i++ {
							l1MultiSink = L1NormSIMD(x)
						}
					} else {
						for i := 0; i < b.N; i++ {
							l1MultiSink = L1NormIncSIMD(x, n, 1)
						}
					}
				})
			}
		}
	}
}
