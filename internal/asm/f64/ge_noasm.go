// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || noasm || gccgo || safe
// +build !amd64 noasm gccgo safe

package f64

// Ger performs the rank-one operation
//
//	A += alpha * x * yᵀ
//
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr) {
	if incX == 1 && incY == 1 {
		x = x[:m]
		y = y[:n]
		for i, xv := range x {
			AxpyUnitary(alpha*xv, y, a[uintptr(i)*lda:uintptr(i)*lda+n])
		}
		return
	}

	var ky, kx uintptr
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
	}
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}

	ix := kx
	for i := 0; i < int(m); i++ {
		AxpyInc(alpha*x[ix], y, a[uintptr(i)*lda:uintptr(i)*lda+n], n, incY, 1, ky, 0)
		ix += incX
	}
}

// gemvN computes
//
//	y = alpha * A * x + beta * y
//
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func gemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(n-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(m-1) * int(incY))
	}

	if incX == 1 && incY == 1 {
		if beta == 0 {
			for i = 0; i < m; i++ {
				y[i] = alpha * DotUnitary(a[lda*i:lda*i+n], x)
			}
			return
		}
		for i = 0; i < m; i++ {
			y[i] = y[i]*beta + alpha*DotUnitary(a[lda*i:lda*i+n], x)
		}
		return
	}
	iy := ky
	if beta == 0 {
		for i = 0; i < m; i++ {
			y[iy] = alpha * DotInc(x, a[lda*i:lda*i+n], n, incX, 1, kx, 0)
			iy += incY
		}
		return
	}
	for i = 0; i < m; i++ {
		y[iy] = y[iy]*beta + alpha*DotInc(x, a[lda*i:lda*i+n], n, incX, 1, kx, 0)
		iy += incY
	}
}

// gemvT computes
//
//	y = alpha * Aᵀ * x + beta * y
//
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func gemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky, i uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
	}
	switch {
	case beta == 0: // beta == 0 is special-cased to memclear
		if incY == 1 {
			for i := range y[:n] {
				y[i] = 0
			}
		} else {
			iy := ky
			for i := 0; i < int(n); i++ {
				y[iy] = 0
				iy += incY
			}
		}
	case int(incY) < 0:
		ScalInc(beta, y, n, uintptr(int(-incY)))
	case incY == 1:
		ScalUnitary(beta, y[:n])
	default:
		ScalInc(beta, y, n, incY)
	}

	if incX == 1 && incY == 1 {
		for i = 0; i < m; i++ {
			AxpyUnitaryTo(y, alpha*x[i], a[lda*i:lda*i+n], y)
		}
		return
	}
	// The AXPY loop is faster for short rows despite its redundant bounds check.
	if n < 32 {
		ix := kx
		for i = 0; i < m; i++ {
			AxpyInc(alpha*x[ix], a[lda*i:lda*i+n], y, n, 1, incY, 0, ky)
			ix += incX
		}
		return
	}
	ix, row := kx, uintptr(0)
	for i = 0; i < m; i++ {
		scale := alpha * x[ix]
		iy := ky
		rowA := a[row : row+n]
		for j := range rowA {
			y[iy] += scale * rowA[j]
			iy += incY
		}
		ix += incX
		row += lda
	}
}
