// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && (!arm64 || !go1.27 || !goexperiment.simd)) || noasm || gccgo || safe

package f64

// GemvN computes y = alpha * A * x + beta * y, where A is an m×n dense matrix.
func GemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	gemvN(m, n, alpha, a, lda, x, incX, beta, y, incY)
}

// GemvT computes y = alpha * Aᵀ * x + beta * y, where A is an m×n dense matrix.
func GemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	gemvT(m, n, alpha, a, lda, x, incX, beta, y, incY)
}
