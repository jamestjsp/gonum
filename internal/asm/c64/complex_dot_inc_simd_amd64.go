// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"unsafe"
)

// Keep native admission and complete recovery in the public entry, avoiding
// the intermediate slice-argument call before the unchanged raw dot leaves.
func DotcIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
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
		var sum complex64
		if n >= 64 {
			sum = complexDotIncEightSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8, true)
		} else {
			sum = complexDotIncUncheckedSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8, true)
		}
		if math.Float32bits(real(sum))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(sum))&0x7f800000 != 0x7f800000 {
			return sum
		}
	}
	// Keep the checked grouping before portable and sequential recovery.
	return complexDotIncCheckedSIMD(x, y, n, incX, incY, ix, iy, true)
}

func DotuIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
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
		var sum complex64
		if n >= 64 {
			sum = complexDotIncEightSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8, false)
		} else {
			sum = complexDotIncUncheckedSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8, false)
		}
		if math.Float32bits(real(sum))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(sum))&0x7f800000 != 0x7f800000 {
			return sum
		}
	}
	// Keep the checked grouping before portable and sequential recovery.
	return complexDotIncCheckedSIMD(x, y, n, incX, incY, ix, iy, false)
}
