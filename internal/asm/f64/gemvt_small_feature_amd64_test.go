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

func TestGemvTSmallFeaturesAndSpans(t *testing.T) {
	enabled := !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
	for n := uintptr(4); n <= 16; n++ {
		a, x, y := make([]float64, n), []float64{1}, make([]float64, n)
		if got := gemvTSmallHardwareSIMD(1, n, 1, a, ^uintptr(0), x, 0, y); got != enabled {
			t.Fatalf("n=%d single row unused huge lda: got=%t want=%t", n, got, enabled)
		}
		for _, bad := range []struct {
			m, n, lda uintptr
			a, x, y   []float64
		}{
			{0, n, n, a, x, y},
			{1, n, n - 1, a, x, y},
			{1, n, n, a[:n-1], x, y},
			{1, n, n, a, x, y[:n-1]},
			{1, n, n, a, nil, y},
			{2, n, ^uintptr(0), a, []float64{1, 1}, y},
			{2, n, n, a, []float64{1, 1}, y},
			{1, n, n, a, x, a},
		} {
			if gemvTSmallHardwareSIMD(bad.m, bad.n, 1, bad.a, bad.lda, bad.x, 0, bad.y) {
				t.Fatalf("accepted invalid/aliased span: n=%d case=%+v", n, bad)
			}
		}
	}
	for _, n := range []uintptr{0, 1, 2, 3, 17, ^uintptr(0)} {
		if gemvTSmallHardwareSIMD(1, n, 1, make([]float64, 32), 32, []float64{1}, 0, make([]float64, 32)) {
			t.Fatalf("accepted unsupported n=%d", n)
		}
	}
}
