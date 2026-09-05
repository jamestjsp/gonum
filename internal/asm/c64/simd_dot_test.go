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

func TestSIMDComplexDotOverflowCancellation(t *testing.T) {
	for _, n := range []int{4, 8, 16, 32, 65, 129} {
		x, y := make([]complex64, n), make([]complex64, n)
		v := complex64(complex(math.Ldexp(1, 62), math.Ldexp(1, 62)))
		for i := range x {
			x[i], y[i] = v, v
		}
		for _, conjugate := range []bool{false, true} {
			var want complex64
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

func TestSIMDComplexStridedDotOverflowCancellation(t *testing.T) {
	for _, n := range []int{4, 8, 16, 32, 65, 129} {
		x, y := make([]complex64, 3*n), make([]complex64, 5*n)
		v := complex64(complex(math.Ldexp(1, 62), math.Ldexp(1, 62)))
		for i := 0; i < n; i++ {
			x[3*i], y[5*i] = v, v
		}
		for _, conjugate := range []bool{false, true} {
			var want complex64
			for i := 0; i < n; i++ {
				v := x[3*i]
				if conjugate {
					v = conj64(v)
				}
				want += v * y[5*i]
			}
			got := DotuIncSIMD(x, y, uintptr(n), 3, 5, 0, 0)
			if conjugate {
				got = DotcIncSIMD(x, y, uintptr(n), 3, 5, 0, 0)
			}
			if got != want {
				t.Fatalf("strided n=%d conjugate=%t: got %v want %v", n, conjugate, got, want)
			}
		}
	}
}

// These finite inputs overflow both the new native grouping and a sequential
// retry. The established portable grouping cancels the large terms first.
func TestSIMDComplexPortableGroupingRecovery(t *testing.T) {
	if simd.Emulated() || simd.VectorBitSize() != 512 {
		t.Skip("512-bit grouping regression")
	}
	m := float32(0.75 * math.MaxFloat32)
	for _, conjugate := range []bool{false, true} {
		n := 32
		x, y := make([]complex64, 2*n), make([]complex64, 2*n)
		for i := 0; i < n; i++ {
			y[2*i] = 1
		}
		for _, i := range []int{0, 1, 4, 5} {
			x[2*i] = complex(m, 0)
			x[2*(i+16)] = complex(-m, 0)
		}
		got := DotuIncSIMD(x, y, uintptr(n), 2, 2, 0, 0)
		if conjugate {
			got = DotcIncSIMD(x, y, uintptr(n), 2, 2, 0, 0)
		}
		if got != 0 {
			t.Errorf("strided conjugate=%t: got %v want 0", conjugate, got)
		}
		pattern := []float32{m, m, 0, 0, m, m, 0, 0, -m, -m, 0, 0, -m, -m, 0, 0}
		x, y = make([]complex64, len(pattern)), make([]complex64, len(pattern))
		for i, v := range pattern {
			x[i], y[i] = complex(v, 0), 1
		}
		got = DotuUnitarySIMD(x, y)
		if conjugate {
			got = DotcUnitarySIMD(x, y)
		}
		if got != 0 {
			t.Errorf("unitary conjugate=%t: got %v want 0", conjugate, got)
		}
	}
}
