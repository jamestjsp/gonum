// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import "unsafe"

// Direct native public entries avoid an intermediate argument-copy frame.
func AxpyIncSIMD(alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		AxpyUnitaryToSIMD(y[iy:iy+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incX == 0 || incY == 0 || !complexIncrementsCompatibleSIMD(x, y, incX, incY, ix, iy) {
		for ; n > 0; n-- {
			y[iy] = alpha*x[ix] + y[iy]
			ix += incX
			iy += incY
		}
		return
	}
	if complexNativeSIMD() {
		if complexStrideInBoundsSIMD(len(x), n, incX, ix) && complexStrideInBoundsSIMD(len(y), n, incY, iy) {
			if incX == incY {
				complexAxpyEqualIncInPlaceUncheckedSIMD(alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8)
			} else {
				complexAxpyIncUncheckedSIMD(unsafe.Pointer(&y[iy]), incY*8, alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8)
			}
		} else {
			complexAxpyIncCheckedSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
		}
		return
	}
	portableAxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func AxpyIncToSIMD(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 && incDst == 1 {
		AxpyUnitaryToSIMD(dst[idst:idst+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incX == 0 || incY == 0 || incDst == 0 || !complexIncrementsCompatibleSIMD(x, dst, incX, incDst, ix, idst) || !complexIncrementsCompatibleSIMD(y, dst, incY, incDst, iy, idst) {
		for ; n > 0; n-- {
			dst[idst] = alpha*x[ix] + y[iy]
			ix += incX
			iy += incY
			idst += incDst
		}
		return
	}

	if complexNativeSIMD() {
		if complexStrideInBoundsSIMD(len(dst), n, incDst, idst) && complexStrideInBoundsSIMD(len(x), n, incX, ix) && complexStrideInBoundsSIMD(len(y), n, incY, iy) {
			if incX == incY && incX == incDst {
				complexAxpyEqualIncUncheckedSIMD(unsafe.Pointer(&dst[idst]), alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8)
			} else {
				complexAxpyIncUncheckedSIMD(unsafe.Pointer(&dst[idst]), incDst*8, alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8)
			}
		} else {
			// Retain ordered writes before panic for invalid streams.
			complexAxpyIncCheckedSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy)
		}
		return
	}

	portableAxpyIncToSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy)
}
