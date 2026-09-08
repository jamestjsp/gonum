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
	if axpyIncPositiveHardwareSIMD(dst, 2, 0, 2, x, y, 4, 2, 2, 0, 0) {
		t.Fatal("positive-span AXPY kernel selected without AVX2")
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

func TestSIMDStridedGERWithoutAVX2(t *testing.T) {
	if archsimd.X86.AVX2() {
		t.Skip("requires AVX2 disabled")
	}
	if supportsAVX2SIMD() {
		t.Fatal("AVX2 tail and short GER helpers enabled without AVX2")
	}
	x := make([]float32, 15)
	y := make([]float32, 31)
	a := make([]float32, 8*16)
	for i := range a {
		a[i] = -1
	}
	if gerPositiveHardwareSIMD(8, 16, 2, x, 2, y, 2, a, 16) {
		t.Fatal("AVX2 GER kernel selected without AVX2")
	}
	for i, v := range a {
		if v != -1 {
			t.Fatalf("declined GER kernel modified a[%d]=%v", i, v)
		}
	}
}

// Native512 may remain selected when AVX2 is disabled independently.
func TestSIMDShortDDOTWithoutAVX2(t *testing.T) {
	if archsimd.X86.AVX2() {
		t.Skip("requires AVX2 disabled")
	}
	for _, n := range []int{7, 15, 16, 31, 32, 64, 4097} {
		if canDdotShortSIMD(n) {
			t.Fatalf("native256 short DDOT selected without AVX2 for n=%d", n)
		}
		x, y := make([]float32, n), make([]float32, n)
		var want float64
		for i := range x {
			x[i], y[i] = float32(i%7-3)/8, float32(i%5-2)/4
			want += float64(x[i]) * float64(y[i])
		}
		if got := DdotUnitarySIMD(x, y); got != want {
			t.Errorf("fallback DDOT n=%d: got %v, want %v", n, got, want)
		}
	}
}
