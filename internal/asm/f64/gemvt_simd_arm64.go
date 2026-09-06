// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

func gemvTShortStridedValid(m, n uintptr, a []float64, lda uintptr, x []float64, incX uintptr, y []float64, incY uintptr) bool {
	if m < 8 || n == 0 || n > 32 || lda < n || int(incX) <= 0 || incY <= 1 || int(incY) <= 0 {
		return false
	}
	aLen, ok := matrixSpan(m, n, lda)
	if !ok || aLen > uintptr(len(a)) {
		return false
	}
	xLen, ok := vectorSpan(m, incX)
	if !ok || xLen > uintptr(len(x)) {
		return false
	}
	yLen, ok := vectorSpan(n, incY)
	if !ok || yLen > uintptr(len(y)) {
		return false
	}
	activeY := y[:yLen]
	return simdMatrixDisjoint(activeY, a[:aLen]) && simdMatrixDisjoint(activeY, x[:xLen])
}

func matrixSpan(rows, cols, stride uintptr) (uintptr, bool) {
	const maxUint = ^uintptr(0)
	if rows == 0 {
		return 0, true
	}
	if stride == 0 || rows-1 > (maxUint-cols)/stride {
		return 0, false
	}
	return (rows-1)*stride + cols, true
}

func vectorSpan(n, inc uintptr) (uintptr, bool) {
	const maxUint = ^uintptr(0)
	if n == 0 {
		return 0, true
	}
	if inc == 0 || n-1 > (maxUint-1)/inc {
		return 0, false
	}
	return (n-1)*inc + 1, true
}

func gemvTShortStrided(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var j uintptr
	for ; j+4 <= n; j += 4 {
		iy0 := j * incY
		iy1 := iy0 + incY
		iy2 := iy1 + incY
		iy3 := iy2 + incY
		var y0, y1, y2, y3 float64
		if beta != 0 {
			y0 = beta * y[iy0]
			y1 = beta * y[iy1]
			y2 = beta * y[iy2]
			y3 = beta * y[iy3]
		}
		var ix, row uintptr
		for i := uintptr(0); i < m; i++ {
			scale := alpha * x[ix]
			y0 += scale * a[row+j]
			y1 += scale * a[row+j+1]
			y2 += scale * a[row+j+2]
			y3 += scale * a[row+j+3]
			ix += incX
			row += lda
		}
		y[iy0] = y0
		y[iy1] = y1
		y[iy2] = y2
		y[iy3] = y3
	}
	for ; j+2 <= n; j += 2 {
		iy0 := j * incY
		iy1 := iy0 + incY
		var y0, y1 float64
		if beta != 0 {
			y0 = beta * y[iy0]
			y1 = beta * y[iy1]
		}
		var ix, row uintptr
		for i := uintptr(0); i < m; i++ {
			scale := alpha * x[ix]
			y0 += scale * a[row+j]
			y1 += scale * a[row+j+1]
			ix += incX
			row += lda
		}
		y[iy0] = y0
		y[iy1] = y1
	}
	for ; j < n; j++ {
		iy := j * incY
		var value float64
		if beta != 0 {
			value = beta * y[iy]
		}
		var ix, row uintptr
		for i := uintptr(0); i < m; i++ {
			scale := alpha * x[ix]
			value += scale * a[row+j]
			ix += incX
			row += lda
		}
		y[iy] = value
	}
}
