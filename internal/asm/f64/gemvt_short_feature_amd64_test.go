// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"simd"
	"simd/archsimd"
	"testing"
)

func TestGemvTEightFeatureAndSpanGuard(t *testing.T) {
	available := !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
	for _, test := range []struct {
		m, lda uintptr
		valid  bool
	}{
		{0, 8, false}, {1, 7, false}, {1, 8, true}, {1, ^uintptr(0), true},
		{2, ^uintptr(0), false}, {2, 8, true}, {2, 16, false},
	} {
		x, a, y := []float64{1, 2}, make([]float64, 16), make([]float64, 8)
		if got := gemvTEightHardwareSIMD(test.m, .5, a, test.lda, x, .25, y); got != (test.valid && available) {
			t.Fatalf("m=%d lda=%d accepted=%t valid=%t available=%t", test.m, test.lda, got, test.valid, available)
		}
	}
}
