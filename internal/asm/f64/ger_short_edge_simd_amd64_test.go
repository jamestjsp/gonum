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

func TestGerEightSingleRowHugeStride(t *testing.T) {
	x := []float64{2, 3}
	y := []float64{1, 1, 1, 1, 1, 1, 1, 1}
	a := make([]float64, 8)
	got := gerEightHardwareSIMD(1, 1, x, y, a, ^uintptr(0))
	enabled := !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
	if got != enabled {
		t.Fatalf("one row: handled=%v enabled=%v", got, enabled)
	}
	if got {
		for i, v := range a {
			if v != 2 {
				t.Fatalf("index=%d got=%g", i, v)
			}
		}
	}
	if gerEightHardwareSIMD(2, 1, x, y, a, ^uintptr(0)) {
		t.Fatal("invalid two-row span accepted")
	}
}
