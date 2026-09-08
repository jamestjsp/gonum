// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"math/big"
	"simd"
	"simd/archsimd"
	"testing"
)

func TestL2NormIncHardwareSIMD(t *testing.T) {
	available := !simd.Emulated() && simd.BroadcastFloat64s(0).Len() >= 4 && archsimd.X86.AVX2() && archsimd.X86.FMA()
	sizes := []int{15, 47, 48, 49, 63, 64, 65, 129, 4097}
	for n := 16; n <= 33; n++ {
		sizes = append(sizes, n)
	}
	for _, n := range sizes {
		for _, inc := range []int{2, 3, 7, 16} {
			for _, exp := range []int{-450, 0, 450} {
				for _, distribution := range []string{"dyadic", "dominant", "product"} {
					x := make([]float64, (n-1)*inc+1)
					for i := range x {
						x[i] = math.NaN()
					}
					for i := 0; i < n; i++ {
						v := 1 + float64(i%31)/64
						if distribution == "dominant" && i != 0 {
							v = 0x1p-27
						} else if distribution == "product" {
							v = 1 + float64(i%63)*0x1p-27
						}
						x[i*inc] = math.Ldexp(v, exp)
					}
					got, ok := l2NormIncHardwareSIMD(x, uintptr(n), uintptr(inc))
					if ok != (available && n >= 16) {
						t.Fatalf("n=%d inc=%d: accepted=%t, hardware available=%t", n, inc, ok, available)
					}
					if ok {
						checkNativeNormULP(t, got, nativeNormReference(x, n, inc))
					}
				}
			}
		}
	}
	for _, tc := range []struct{ n, inc uintptr }{{0, 2}, {32, 0}, {32, 1}, {32, ^uintptr(0)}, {32, 100}, {^uintptr(0), 2}} {
		if _, ok := l2NormIncHardwareSIMD(make([]float64, 65), tc.n, tc.inc); ok {
			t.Fatalf("accepted unsupported span n=%d inc=%d", tc.n, tc.inc)
		}
	}
	if !available {
		return
	}
	const n, inc = 33, 3
	for _, special := range []float64{0, math.SmallestNonzeroFloat64, 0x1p-600, 0x1p-500, 0x1p500, 0x1p1000, math.Inf(1), math.Inf(-1), math.NaN()} {
		x := make([]float64, (n-1)*inc+1)
		for i := 0; i < n; i++ {
			x[i*inc] = special
		}
		got, ok := l2NormIncHardwareSIMD(x, n, inc)
		if !ok {
			t.Fatal("ordinary valid geometry rejected")
		}
		if math.IsNaN(special) || math.IsInf(special, 0) {
			checkSIMDNorm(t, got, math.Abs(special))
		} else {
			// Exceptional magnitudes retain the existing scaled recurrence and
			// its tolerance; ordinary compensated cases above retain 1 ULP.
			checkSIMDNorm(t, got, nativeNormReference(x, n, inc))
		}
	}
}

func nativeNormReference(x []float64, n, inc int) float64 {
	sum, value, product := new(big.Float).SetPrec(256), new(big.Float).SetPrec(256), new(big.Float).SetPrec(256)
	for i := 0; i < n; i++ {
		value.SetFloat64(x[i*inc])
		sum.Add(sum, product.Mul(value, value))
	}
	want, _ := sum.Sqrt(sum).Float64()
	return want
}

func checkNativeNormULP(t *testing.T, got, want float64) {
	t.Helper()
	a, b := math.Float64bits(got), math.Float64bits(want)
	if a < b {
		a, b = b, a
	}
	if math.IsNaN(got) || math.IsInf(got, 0) || a-b > 1 {
		t.Fatalf("native norm: got %.17g, want %.17g; error %d ULP", got, want, a-b)
	}
}

// Run under -race: inactive stride gaps can be owned by another caller.
func TestL2NormIncHardwareIndependentGaps(t *testing.T) {
	const n, inc = 65, 3
	x := make([]float64, (n-1)*inc+1)
	for i := 0; i < n; i++ {
		x[i*inc] = 1
	}
	if _, ok := l2NormIncHardwareSIMD(x, n, inc); !ok {
		t.Skip("native norm unavailable")
	}
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		for {
			select {
			case <-stop:
				return
			default:
				for i := 0; i < n-1; i++ {
					x[i*inc+1]++
				}
			}
		}
	}()
	var got float64
	for i := 0; i < 1000; i++ {
		got, _ = l2NormIncHardwareSIMD(x, n, inc)
	}
	close(stop)
	<-done
	checkNativeNormULP(t, got, math.Sqrt(n))
}
