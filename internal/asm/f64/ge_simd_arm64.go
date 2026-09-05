// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

// GemvN computes y = alpha * A * x + beta * y, where A is an m×n dense matrix.
func GemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	if m >= 8 && n >= 8 && incX == 1 && incY == 1 && lda >= n && simdMatrixDisjoint(y, x) && simdMatrixDisjoint(y, a) {
		GemvNSIMD(m, n, alpha, a, lda, x, incX, beta, y, incY)
		return
	}
	gemvN(m, n, alpha, a, lda, x, incX, beta, y, incY)
}

// GemvT computes y = alpha * Aᵀ * x + beta * y, where A is an m×n dense matrix.
func GemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	if m >= 8 && n >= 8 && (m >= 32 || n <= 16) && incX == 1 && incY == 1 && lda >= n && simdMatrixDisjoint(y, x) && simdMatrixDisjoint(y, a) {
		GemvTSIMD(m, n, alpha, a, lda, x, incX, beta, y, incY)
		return
	}
	gemvT(m, n, alpha, a, lda, x, incX, beta, y, incY)
}
