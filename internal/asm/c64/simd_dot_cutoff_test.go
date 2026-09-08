// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd"
	"testing"
)

func TestSIMDComplexDotCutoffBoundaries(t *testing.T) {
	for _, n := range []int{127, 128, 129, 255, 256, 257, 511, 512, 513} {
		x, y := make([]complex64, n), make([]complex64, n)
		var wantc, wantu complex128
		for i := range x {
			x[i] = complex(float32(i%11-5)/16, float32(i%7-3)/8)
			y[i] = complex(float32(i%13-6)/8, float32(i%5-2)/16)
			xv, yv := complex128(x[i]), complex128(y[i])
			wantu += xv * yv
			wantc += complex(real(xv), -imag(xv)) * yv
		}
		// Every product and sum is exactly representable for these dyadic inputs.
		if got := DotuUnitarySIMD(x, y); complex128(got) != wantu {
			t.Errorf("Dotu n=%d got %v want %v", n, got, wantu)
		}
		if got := DotcUnitarySIMD(x, y); complex128(got) != wantc {
			t.Errorf("Dotc n=%d got %v want %v", n, got, wantc)
		}
		if !complexNativeSIMD() || simd.VectorBitSize() < 256 {
			continue
		}
		clear(x)
		for i := range y {
			y[i] = 1
		}
		m := float32(0.75 * math.MaxFloat32)
		x[0], x[4], x[8], x[12] = complex(m, 0), complex(-m, 0), complex(m, 0), complex(-m, 0)
		for _, conjugate := range []bool{false, true} {
			raw := complexDotShortWideSIMD(x, y, conjugate)
			if math.Float32bits(real(raw))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(raw))&0x7f800000 != 0x7f800000 {
				t.Fatalf("n=%d conjugate=%t: wide fixture did not overflow: %v", n, conjugate, raw)
			}
			want := portableDotUnitarySIMD(x, y, conjugate)
			if want != 0 {
				t.Fatalf("n=%d conjugate=%t: original recovery returned %v", n, conjugate, want)
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
