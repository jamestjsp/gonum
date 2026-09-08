// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"simd/archsimd"
	"testing"
)

// Run this in separate processes with widths 0/128/256/512 and independently
// disabled AVX/AVX2. The cache must preserve the original admission decisions,
// including the native512 configuration with AVX2 disabled.
func TestSIMDCachedEligibility(t *testing.T) {
	for n := 0; n <= 260; n++ {
		wantReduction := n < 256 && !simd.Emulated() && archsimd.X86.AVX() &&
			(n < 32 || simd.VectorBitSize() >= 256 && archsimd.X86.AVX2())
		if got := n < nativeReductionLimitSIMD; got != wantReduction {
			t.Errorf("reduction n=%d: admitted=%t, want %t", n, got, wantReduction)
		}
		wantDdot := n < 32 && n&7 != 0 && !simd.Emulated() &&
			simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
		if got := canDdotShortSIMD(n); got != wantDdot {
			t.Errorf("ddot n=%d: admitted=%t, want %t", n, got, wantDdot)
		}
	}
}
