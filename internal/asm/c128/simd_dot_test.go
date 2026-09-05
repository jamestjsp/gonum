// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math"
	"testing"
)

func TestSIMDComplexDotOverflowCancellation(t *testing.T) {
	for _, n := range []int{4, 8, 16, 32, 65, 129} {
		x, y := make([]complex128, n), make([]complex128, n)
		v := complex128(complex(math.Ldexp(1, 510), math.Ldexp(1, 510)))
		for i := range x {
			x[i], y[i] = v, v
		}
		for _, conjugate := range []bool{false, true} {
			var want complex128
			for i, v := range x {
				if conjugate {
					v = complex(real(v), -imag(v))
				}
				want += v * y[i]
			}
			got := DotuUnitarySIMD(x, y)
			if conjugate {
				got = DotcUnitarySIMD(x, y)
			}
			if got != want {
				t.Errorf("n=%d conjugate=%t: got %v want %v", n, conjugate, got, want)
			}
		}
	}
}
