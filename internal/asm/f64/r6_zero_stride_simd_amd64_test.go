// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"testing"
)

func TestR6L1NormIncSIMDZeroStride(t *testing.T) {
	x := []float64{-3}
	got := L1NormIncSIMD(x, 2, 0)
	t.Logf("R6_ZERO route=L1NormIncSIMD n=2 inc=0 len=%d cap=%d got_bits=%016x documented_bits=%016x width=%d emulated=%t avx=%t avx2=%t", len(x), cap(x), math.Float64bits(got), uint64(0), simd.VectorBitSize(), simd.Emulated(), archsimd.X86.AVX(), archsimd.X86.AVX2())
	if math.Float64bits(x[0]) != math.Float64bits(-3) {
		t.Fatal("input changed")
	}
	if math.Float64bits(got) != 0 {
		t.Fatalf("documented zero-stride loop returns +0; got %g", got)
	}
}
