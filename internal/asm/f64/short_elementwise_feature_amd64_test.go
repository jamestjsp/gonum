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

func TestSIMDShortElementwiseFeatureGate(t *testing.T) {
	// BroadcastFloat64x2 and BroadcastUint64x2 require AVX2, even though
	// the loads and floating-point arithmetic need only AVX.
	if !archsimd.X86.AVX2() && shortElementwiseHardwareSIMD(31) {
		t.Fatal("short native elementwise helper enabled without AVX2")
	}
}

func TestSIMDShortScalWidthGate(t *testing.T) {
	for _, n := range []int{0, 1, 31, 63, 64} {
		want := n < 64 && !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
		if got := shortScalHardwareSIMD(n); got != want {
			t.Fatalf("n=%d width=%d emulated=%t AVX2=%t: native scaling enabled=%t want=%t", n, simd.VectorBitSize(), simd.Emulated(), archsimd.X86.AVX2(), got, want)
		}
	}
}
