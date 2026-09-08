// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math"
	"simd"
	"unsafe"
)

// Native public entries avoid an intermediate argument-copy frame when callers
// hold the candidate as a function value. Keep complete checked-group recovery.
func DotcIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return portableDotUnitarySIMD(x[ix:ix+n], y[iy:iy+n], true)
	}
	if !complexNativeSIMD() {
		return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, true)
	}
	if complexStrideInBoundsSIMD(len(x), n, incX, ix) && complexStrideInBoundsSIMD(len(y), n, incY, iy) {
		var sum complex128
		if simd.VectorBitSize() >= 256 {
			sum = complexDotIncPairSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*16, incY*16, true)
		} else {
			sum = complexDotIncUncheckedSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*16, incY*16, true)
		}
		if math.Float64bits(real(sum))&0x7ff0000000000000 != 0x7ff0000000000000 && math.Float64bits(imag(sum))&0x7ff0000000000000 != 0x7ff0000000000000 {
			return sum
		}
	}
	// Keep the established grouping and its portable/sequential recovery.
	return complexDotIncCheckedSIMD(x, y, n, incX, incY, ix, iy, true)
}

func DotuIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return portableDotUnitarySIMD(x[ix:ix+n], y[iy:iy+n], false)
	}
	if !complexNativeSIMD() {
		return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, false)
	}
	if complexStrideInBoundsSIMD(len(x), n, incX, ix) && complexStrideInBoundsSIMD(len(y), n, incY, iy) {
		var sum complex128
		if simd.VectorBitSize() >= 256 {
			sum = complexDotIncPairSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*16, incY*16, false)
		} else {
			sum = complexDotIncUncheckedSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*16, incY*16, false)
		}
		if math.Float64bits(real(sum))&0x7ff0000000000000 != 0x7ff0000000000000 && math.Float64bits(imag(sum))&0x7ff0000000000000 != 0x7ff0000000000000 {
			return sum
		}
	}
	// Keep the established grouping and its portable/sequential recovery.
	return complexDotIncCheckedSIMD(x, y, n, incX, incY, ix, iy, false)
}
