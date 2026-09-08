// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math"
	"testing"
)

func TestSIMDComplexDscalTinyIEEE(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	values := []complex128{
		complex(0, negativeZero), complex(negativeZero, 0),
		1 - 2i, -3 + 0.5i,
		complex(math.MaxFloat64, -math.MaxFloat64),
		complex(math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64),
		complex(math.Inf(1), math.Inf(-1)),
		complex(math.NaN(), 1), complex(-1, math.NaN()),
	}
	alphas := []float64{0, negativeZero, 1, -1, 0.5, -2, math.MaxFloat64, math.SmallestNonzeroFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	same := func(got, want float64) bool {
		return math.Float64bits(got) == math.Float64bits(want) || math.IsNaN(got) && math.IsNaN(want)
	}
	for _, n := range []int{0, 1, 2, 3, 4, 8} {
		for _, alpha := range alphas {
			for start := range values {
				backing := make([]complex128, n+4)
				for i := range backing {
					backing[i] = 17 + 19i
				}
				x := backing[2 : 2+n : 2+n]
				want := make([]complex128, n)
				for i := range x {
					x[i] = values[(start+i)%len(values)]
					want[i] = complex(real(x[i])*alpha, imag(x[i])*alpha)
				}
				DscalUnitarySIMD(alpha, x)
				for i, value := range want {
					if !same(real(x[i]), real(value)) || !same(imag(x[i]), imag(value)) {
						t.Errorf("n=%d alpha=%v start=%d i=%d got=%v want=%v", n, alpha, start, i, x[i], value)
					}
				}
				for _, i := range []int{0, 1, n + 2, n + 3} {
					if backing[i] != 17+19i {
						t.Errorf("n=%d wrote outside the slice at index%d", n, i)
					}
				}
			}
		}
	}
}
