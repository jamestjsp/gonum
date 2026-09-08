// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

func DotcIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return portableDotUnitarySIMD(x[ix:ix+n], y[iy:iy+n], true)
	}
	if complexNativeSIMD() {
		return complexDotIncNativeSIMD(x, y, n, incX, incY, ix, iy, true)
	}
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, true)
}

func DotuIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return portableDotUnitarySIMD(x[ix:ix+n], y[iy:iy+n], false)
	}
	if complexNativeSIMD() {
		return complexDotIncNativeSIMD(x, y, n, incX, incY, ix, iy, false)
	}
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, false)
}
