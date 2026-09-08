// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestNormShortNativeBoundaries(t *testing.T) {
	for n := 0; n <= 129; n++ {
		for _, exp := range []int{-450, 0, 450} {
			x, y, d := make([]float64, n), make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = math.Ldexp(1+float64(i%63)*0x1p-27, exp)
				y[i] = -0.125 * x[i]
				d[i] = x[i] - y[i]
			}
			checkNativeNormULP(t, L2NormUnitarySIMD(x), nativeNormReference(x, n, 1))
			checkNativeNormULP(t, L2DistanceUnitarySIMD(x, y), nativeNormReference(d, n, 1))
		}
		for _, v := range []float64{0, math.SmallestNonzeroFloat64, 0x1p-600, 0x1p1000, math.Inf(1), math.NaN()} {
			x, y := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = v
			}
			checkSIMDNorm(t, L2NormUnitarySIMD(x), l2NormUnitaryScalar(x))
			checkSIMDNorm(t, L2DistanceUnitarySIMD(x, y), l2DistanceUnitaryScalar(x, y))
		}
	}
}
