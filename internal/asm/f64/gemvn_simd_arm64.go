// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

func gemvNShortStridedValid(m, n uintptr, a []float64, lda uintptr, x []float64, incX uintptr, y []float64, incY uintptr) bool {
	if m < 4 || m > 32 || n < 8 || lda < n || incX != 1 || incY <= 1 || int(incY) <= 0 {
		return false
	}
	aLen, ok := matrixSpan(m, n, lda)
	if !ok || aLen > uintptr(len(a)) {
		return false
	}
	xLen, ok := vectorSpan(n, incX)
	if !ok || xLen > uintptr(len(x)) {
		return false
	}
	yLen, ok := vectorSpan(m, incY)
	if !ok || yLen > uintptr(len(y)) {
		return false
	}
	activeY := y[:yLen]
	return simdMatrixDisjoint(activeY, a[:aLen]) && simdMatrixDisjoint(activeY, x[:xLen])
}

func gemvNShortStrided(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, beta float64, y []float64, incY uintptr) {
	x = x[:n:n]
	var row uintptr
	for ; row+4 <= m; row += 4 {
		a0 := a[row*lda : row*lda+n : row*lda+n]
		a1 := a[(row+1)*lda : (row+1)*lda+n : (row+1)*lda+n]
		a2 := a[(row+2)*lda : (row+2)*lda+n : (row+2)*lda+n]
		a3 := a[(row+3)*lda : (row+3)*lda+n : (row+3)*lda+n]
		var s0, s1, s2, s3 float64
		for col, xv := range x {
			s0 += a0[col] * xv
			s1 += a1[col] * xv
			s2 += a2[col] * xv
			s3 += a3[col] * xv
		}
		iy := row * incY
		if beta == 0 {
			y[iy] = alpha * s0
			y[iy+incY] = alpha * s1
			y[iy+2*incY] = alpha * s2
			y[iy+3*incY] = alpha * s3
		} else {
			y[iy] = y[iy]*beta + alpha*s0
			y[iy+incY] = y[iy+incY]*beta + alpha*s1
			y[iy+2*incY] = y[iy+2*incY]*beta + alpha*s2
			y[iy+3*incY] = y[iy+3*incY]*beta + alpha*s3
		}
	}
	for ; row+2 <= m; row += 2 {
		a0 := a[row*lda : row*lda+n : row*lda+n]
		a1 := a[(row+1)*lda : (row+1)*lda+n : (row+1)*lda+n]
		var s0, s1 float64
		for col, xv := range x {
			s0 += a0[col] * xv
			s1 += a1[col] * xv
		}
		iy := row * incY
		if beta == 0 {
			y[iy] = alpha * s0
			y[iy+incY] = alpha * s1
		} else {
			y[iy] = y[iy]*beta + alpha*s0
			y[iy+incY] = y[iy+incY]*beta + alpha*s1
		}
	}
	for ; row < m; row++ {
		arow := a[row*lda : row*lda+n : row*lda+n]
		var sum float64
		for col, xv := range x {
			sum += arow[col] * xv
		}
		iy := row * incY
		if beta == 0 {
			y[iy] = alpha * sum
		} else {
			y[iy] = y[iy]*beta + alpha*sum
		}
	}
}
