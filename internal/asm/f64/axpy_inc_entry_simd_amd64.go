// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import "unsafe"

func AxpyIncSIMD(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		AxpyUnitarySIMD(alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incX == 0 || incY == 0 || !simdSlicesCompatible(x, y) || unsafe.SliceData(x) == unsafe.SliceData(y) {
		for ; n > 0; n-- {
			y[iy] += alpha * x[ix]
			ix += incX
			iy += incY
		}
		return
	}

	if simdPositiveSpan(len(x), n, incX, ix) && simdPositiveSpan(len(y), n, incY, iy) {
		axpyIncPositiveSIMD(alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8)
		return
	}

	// Direct scalar operations avoid packing and scattering sparse lanes.
	// The separate path above preserves dependencies when inputs overlap.
	for ; n >= 4; n -= 4 {
		y[iy] += alpha * x[ix]
		y[iy+incY] += alpha * x[ix+incX]
		y[iy+2*incY] += alpha * x[ix+2*incX]
		y[iy+3*incY] += alpha * x[ix+3*incX]
		ix += 4 * incX
		iy += 4 * incY
	}
	for ; n > 0; n-- {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func AxpyIncToSIMD(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incDst == 1 && incX == 1 && incY == 1 {
		AxpyUnitaryToSIMD(dst[idst:idst+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incDst == 0 || incX == 0 || incY == 0 || !simdSlicesCompatible(dst, x) || !simdSlicesCompatible(dst, y) || unsafe.SliceData(dst) == unsafe.SliceData(x) || unsafe.SliceData(dst) == unsafe.SliceData(y) {
		for ; n > 0; n-- {
			dst[idst] = alpha*x[ix] + y[iy]
			idst += incDst
			ix += incX
			iy += incY
		}
		return
	}

	if simdPositiveSpan(len(x), n, incX, ix) && simdPositiveSpan(len(y), n, incY, iy) && simdPositiveSpan(len(dst), n, incDst, idst) {
		axpyIncToPositiveSIMD(alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), unsafe.Pointer(&dst[idst]), n, incX*8, incY*8, incDst*8)
		return
	}

	for ; n >= 4; n -= 4 {
		dst[idst] = alpha*x[ix] + y[iy]
		dst[idst+incDst] = alpha*x[ix+incX] + y[iy+incY]
		dst[idst+2*incDst] = alpha*x[ix+2*incX] + y[iy+2*incY]
		dst[idst+3*incDst] = alpha*x[ix+3*incX] + y[iy+3*incY]
		ix += 4 * incX
		iy += 4 * incY
		idst += 4 * incDst
	}
	for ; n > 0; n-- {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}
