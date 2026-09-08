// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd/archsimd"
	"testing"
)

// Exercise this fallback on AVX2 hosts with GODEBUG=simd=128,cpu.avx2=off.
// Portable SIMD128 needs only AVX, while the native leaves also use AVX2.
func TestSIMDComplexNativeRequiresAVX2(t *testing.T) {
	if archsimd.X86.AVX2() {
		t.Skip("AVX2 is available")
	}
	if complexNativeSIMD() {
		t.Fatal("native complex kernels selected without AVX2")
	}
}

// Native short accumulation cancels each product before adding the next. Raw
// components, the wider portable grouping, and a sequential sum can all
// overflow this finite sequence, so recovery must retain the native grouping.
func TestSIMDComplexShortComponentRecovery(t *testing.T) {
	if !complexNativeSIMD() {
		t.Skip("native complex kernel unavailable")
	}
	const n = 16
	m := float32(0.4 * math.MaxFloat32)
	for _, conjugate := range []bool{false, true} {
		x, y := make([]complex64, n), make([]complex64, n)
		for i := range x {
			sign := float32(1)
			if i/4%2 != 0 {
				sign = -1
			}
			x[i] = complex(m, sign*m)
			if conjugate {
				sign = -sign
			}
			y[i] = complex(1, sign)
		}
		got := DotuUnitarySIMD(x, y)
		if conjugate {
			got = DotcUnitarySIMD(x, y)
		}
		if got != 0 {
			t.Errorf("conjugate=%t: got %v want 0", conjugate, got)
		}
	}
}
