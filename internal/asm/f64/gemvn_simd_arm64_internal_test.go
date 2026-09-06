// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import "testing"

func TestGemvNShortStridedSpanOverflow(t *testing.T) {
	const maxUint = ^uintptr(0)
	a, x, y := make([]float64, 32), make([]float64, 8), make([]float64, 7)
	if !gemvNShortStridedValid(4, 8, a, 8, x, 1, y, 2) {
		t.Fatal("valid boundary rejected")
	}
	for _, tc := range []struct{ m, n, lda, incX, incY uintptr }{
		{4, 8, maxUint, 1, 2},
		{4, 8, 8, 1, maxUint / 2},
		{4, maxUint, maxUint, 1, 2},
		{maxUint, 8, 8, 1, 2},
		{4, 8, 8, maxUint, 2},
		{4, 8, 8, 1, maxUint},
	} {
		if gemvNShortStridedValid(tc.m, tc.n, a, tc.lda, x, tc.incX, y, tc.incY) {
			t.Errorf("invalid spans accepted: %+v", tc)
		}
	}
}
