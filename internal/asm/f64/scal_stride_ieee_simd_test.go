// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestSIMDScalStrideIEEE(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -1, 0.3, math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, math.MaxFloat64, -math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	length := func(n, inc int) int {
		if n == 0 {
			return 0
		}
		return (n-1)*inc + 1
	}
	equal := func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
	}
	for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 31, 32, 33, 63, 64, 65, 66, 127} {
		for _, incX := range []int{1, 2, 3, 7} {
			for _, incDst := range []int{1, 2, 3, 7} {
				for _, alpha := range values {
					x := make([]float64, length(n, incX))
					for i := range x {
						x[i] = math.NaN()
					}
					for i := 0; i < n; i++ {
						x[i*incX] = values[(i+n)%len(values)]
					}
					inPlace := append([]float64(nil), x...)
					ScalIncSIMD(alpha, inPlace, uintptr(n), uintptr(incX))
					for i, value := range inPlace {
						want := x[i]
						if i%incX == 0 {
							want *= alpha
						}
						if !equal(value, want) {
							t.Fatalf("Scal n=%d inc=%d alpha=%g index=%d got=%g want=%g", n, incX, alpha, i, value, want)
						}
					}
					dst := make([]float64, length(n, incDst)+2)
					for i := range dst {
						dst[i] = -19
					}
					ScalIncToSIMD(dst[1:len(dst)-1], uintptr(incDst), alpha, x, uintptr(n), uintptr(incX))
					for i, value := range dst {
						want := -19.0
						if i >= 1 && i < len(dst)-1 && (i-1)%incDst == 0 {
							want = alpha * x[(i-1)/incDst*incX]
						}
						if !equal(value, want) {
							t.Fatalf("ScalTo n=%d incX=%d incDst=%d alpha=%g index=%d got=%g want=%g", n, incX, incDst, alpha, i, value, want)
						}
					}
				}
			}
		}
	}
}
