// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"simd"
	"simd/archsimd"
	"testing"
)

// Run this in separate processes with widths 0/128/256/512 and independently
// disabled AVX/AVX2. The two admission decisions remain independent because
// startup overrides do not require AVX and AVX2 flags to agree.
func TestSIMDCachedEligibility(t *testing.T) {
	for n := 0; n <= 260; n++ {
		wantShort := n < 32 && !simd.Emulated() && archsimd.X86.AVX()
		if got := n < nativeShortReductionLimitSIMD; got != wantShort {
			t.Errorf("short n=%d: admitted=%t, want %t", n, got, wantShort)
		}
		wantMedium := n < 128 && !simd.Emulated() &&
			simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
		if got := n < nativeMediumSumLimitSIMD; got != wantMedium {
			t.Errorf("medium n=%d: admitted=%t, want %t", n, got, wantMedium)
		}
	}
}
