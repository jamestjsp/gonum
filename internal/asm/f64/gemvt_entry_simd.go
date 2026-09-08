// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func GemvTSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	if n >= 4 && n <= 16 && incX == 1 && incY == 1 {
		if n == 8 {
			if gemvTEightHardwareSIMD(m, alpha, a, lda, x, beta, y) {
				return
			}
		} else if gemvTSmallHardwareSIMD(m, n, alpha, a, lda, x, beta, y) {
			return
		}
	}
	gemvTPortableSIMD(m, n, alpha, a, lda, x, incX, beta, y, incY)
}
