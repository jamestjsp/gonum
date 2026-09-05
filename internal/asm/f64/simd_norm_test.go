// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"math/big"
	"math/rand/v2"
	"testing"
)

func TestSIMDNormExtremeMagnitudes(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257} {
		for _, exp := range []int{-1074, -1070, -1022, -600, -500, -486, -485, -484, -100, 0, 100, 500, 510, 511, 512, 1000} {
			x, y := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = math.Ldexp(float64(rng.IntN(9)-4), exp)
				y[i] = math.Ldexp(float64(rng.IntN(9)-4), exp)
			}
			normSum, distSum := new(big.Float).SetPrec(256), new(big.Float).SetPrec(256)
			for i, v := range x {
				val := new(big.Float).SetPrec(256).SetFloat64(v)
				normSum.Add(normSum, new(big.Float).SetPrec(256).Mul(val, val))
				val.SetFloat64(v - y[i])
				distSum.Add(distSum, new(big.Float).SetPrec(256).Mul(val, val))
			}
			norm, _ := normSum.Sqrt(normSum).Float64()
			distance, _ := distSum.Sqrt(distSum).Float64()
			checkSIMDNorm(t, L2NormUnitarySIMD(x), norm)
			checkSIMDNorm(t, L2DistanceUnitarySIMD(x, y), distance)
			for _, inc := range []int{1, 2, 3, 7} {
				strided := make([]float64, n*inc)
				for i := range strided {
					strided[i] = math.NaN()
				}
				for i, v := range x {
					strided[i*inc] = v
				}
				checkSIMDNorm(t, L2NormIncSIMD(strided, uintptr(n), uintptr(inc)), norm)
			}
		}
	}
	for _, n := range []int{1, 3, 8, 17, 33, 65} {
		for special := 0; special < n; special++ {
			for _, v := range []float64{math.Inf(1), math.Inf(-1), math.NaN()} {
				x, zero := make([]float64, n), make([]float64, n)
				for i := range x {
					x[i] = float64(i + 1)
				}
				x[special] = v
				want := math.Abs(v)
				checkSIMDNorm(t, L2NormUnitarySIMD(x), want)
				checkSIMDNorm(t, L2DistanceUnitarySIMD(x, zero), want)
				checkSIMDNorm(t, L2NormIncSIMD(x, uintptr(n), 1), want)
				xs := make([]float64, 2*n)
				for i, v := range x {
					xs[2*i] = v
				}
				checkSIMDNorm(t, L2NormIncSIMD(xs, uintptr(n), 2), want)
				if n > 1 {
					x[(special+1)%n] = math.NaN()
					checkSIMDNorm(t, L2NormUnitarySIMD(x), math.NaN())
					checkSIMDNorm(t, L2DistanceUnitarySIMD(x, zero), math.NaN())
				}
			}
		}
	}
}

func checkSIMDNorm(t *testing.T, got, want float64) {
	t.Helper()
	if got == want || math.IsNaN(got) && math.IsNaN(want) {
		return
	}
	// The 256-bit reference sums products before rounding the final square root.
	// Allow floating-point summation error and final subnormal rounding.
	if math.IsNaN(want) || math.IsInf(want, 0) || math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 3e-14*math.Abs(want)+2*math.SmallestNonzeroFloat64 {
		t.Fatalf("norm: got %g, want %g", got, want)
	}
}

func TestSIMDLinfNaNsAndTails(t *testing.T) {
	for n := 0; n < 70; n++ {
		for pos := -1; pos < n; pos++ {
			x, y := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = math.NaN()
			}
			if pos >= 0 {
				x[pos] = 7
			}
			want := math.NaN()
			if n == 0 {
				want = 0
			} else if pos >= 0 {
				want = 7
			}
			checkSIMDNorm(t, LinfDistSIMD(x, y), want)
		}
	}
}
