// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import "testing"

func TestGemvTShortStridedSpanOverflow(t *testing.T) {
	const maxInt = ^uintptr(0) >> 1
	a := make([]float64, 9*2)
	x := make([]float64, 9)
	y := make([]float64, 4)
	for _, tc := range []struct {
		name                  string
		m, n, lda, incX, incY uintptr
	}{
		{name: "m", m: maxInt, n: 2, lda: 2, incX: 1, incY: 2},
		{name: "lda", m: 9, n: 2, lda: maxInt, incX: 1, incY: 2},
		{name: "incx", m: 9, n: 2, lda: 2, incX: maxInt, incY: 2},
		{name: "incy", m: 9, n: 2, lda: 2, incX: 1, incY: maxInt},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if gemvTShortStridedValid(tc.m, tc.n, a, tc.lda, x, tc.incX, y, tc.incY) {
				t.Fatal("overflowing geometry accepted")
			}
		})
	}
}
