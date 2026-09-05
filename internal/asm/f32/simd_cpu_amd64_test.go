// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd/archsimd"
	"testing"
)

// Run with GODEBUG=simd=128,cpu.avx2=off to exercise the AVX-only fallback.
func TestSIMDStridedAXPYWithoutAVX2(t *testing.T) {
	if archsimd.X86.AVX2() {
		t.Skip("requires AVX2 disabled")
	}
	x := []float32{1, 0, 2, 0, 3, 0, 4}
	y := []float32{5, 0, 6, 0, 7, 0, 8}
	dst := []float32{-1, -1, -1, -1, -1, -1, -1}
	if axpyIncHardwareSIMD(dst, 2, 0, 2, x, y, 4, 2, 2, 0, 0) {
		t.Fatal("AVX2 broadcast kernel selected without AVX2")
	}
	for i, v := range dst {
		if v != -1 {
			t.Fatalf("declined kernel modified dst[%d]=%v", i, v)
		}
	}
	AxpyIncToSIMD(dst, 2, 0, 2, x, y, 4, 2, 2, 0, 0)
	want := []float32{7, -1, 10, -1, 13, -1, 16}
	for i, v := range dst {
		if v != want[i] {
			t.Errorf("AVX-only fallback dst[%d]=%v, want %v", i, v, want[i])
		}
	}
}
