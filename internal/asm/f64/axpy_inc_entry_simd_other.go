// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func AxpyIncSIMD(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	axpyIncPortableSIMD(alpha, x, y, n, incX, incY, ix, iy)
}
func AxpyIncToSIMD(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	axpyIncToPortableSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy)
}
