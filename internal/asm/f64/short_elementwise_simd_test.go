// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func sameShortElementwiseFloat(got, want float64) bool {
	return math.Float64bits(got) == math.Float64bits(want) || math.IsNaN(got) && math.IsNaN(want)
}

func TestSIMDShortElementwiseIEEE(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -1, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, math.MaxFloat64, -math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	for n := 0; n <= 65; n++ {
		for offset := 0; offset < 4; offset++ {
			backing, y := make([]float64, n+offset), make([]float64, n)
			x := backing[offset:]
			for i := range x {
				x[i], y[i] = values[(i+offset)%len(values)], values[(3*i+offset+1)%len(values)]
			}
			want := math.NaN()
			if n == 0 {
				want = 0
			}
			for i, v := range x {
				d := math.Abs(y[i] - v)
				if d > want || math.IsNaN(want) {
					want = d
				}
			}
			if got := LinfDistSIMD(x, y); !sameShortElementwiseFloat(got, want) {
				t.Fatalf("Linf n=%d offset=%d: got %g want %g", n, offset, got, want)
			}
			for _, alpha := range []float64{0, math.Copysign(0, -1), 1, -1, 0.3, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.NaN()} {
				inPlace := append([]float64(nil), x...)
				dst := make([]float64, n+2)
				dst[n], dst[n+1] = 17, 19
				ScalUnitarySIMD(alpha, inPlace)
				ScalUnitaryToSIMD(dst, alpha, x)
				for i, v := range x {
					want := alpha * v
					if !sameShortElementwiseFloat(inPlace[i], want) || !sameShortElementwiseFloat(dst[i], want) {
						t.Fatalf("Scal n=%d offset=%d alpha=%g index=%d: inPlace=%g dst=%g want %g", n, offset, alpha, i, inPlace[i], dst[i], want)
					}
				}
				if dst[n] != 17 || dst[n+1] != 19 {
					t.Fatalf("ScalTo overwrote trailing elements n=%d", n)
				}
			}
		}
	}
}

func TestSIMDShortScalOverlap(t *testing.T) {
	for n := 0; n <= 65; n++ {
		for _, shift := range []int{-2, -1, 0, 1, 2} {
			got := make([]float64, n+4)
			for i := range got {
				got[i] = float64(i%13) - 5.25
			}
			want := append([]float64(nil), got...)
			for i := 0; i < n; i++ {
				want[2+shift+i] = -0.3 * want[2+i]
			}
			ScalUnitaryToSIMD(got[2+shift:2+shift+n], -0.3, got[2:2+n])
			for i := range want {
				if !sameShortElementwiseFloat(got[i], want[i]) {
					t.Fatalf("n=%d shift=%d element=%d: got %g want %g", n, shift, i, got[i], want[i])
				}
			}
		}
	}
}

func TestSIMDShortLinfFiniteOverflow(t *testing.T) {
	for n := 1; n <= 65; n++ {
		x, y := make([]float64, n), make([]float64, n)
		for i := range x {
			x[i] = 0.75 * math.MaxFloat64
		}
		// The NaN detector's nonnegative sum may overflow, but the maximum is
		// still finite. It must not corrupt the result or require scaled norms.
		if got := LinfDistSIMD(x, y); got != x[0] {
			t.Fatalf("finite n=%d: got %g want %g", n, got, x[0])
		}
		y[n-1] = -x[n-1]
		if got := LinfDistSIMD(x, y); !math.IsInf(got, 1) {
			t.Fatalf("overflow n=%d: got %g want +Inf", n, got)
		}
	}
}
