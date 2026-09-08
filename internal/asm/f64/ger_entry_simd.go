// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func GerSIMD(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr) {
	if n == 8 && incX == 1 && incY == 1 && gerEightHardwareSIMD(m, alpha, x, y, a, lda) {
		return
	}
	if gerTiledHardwareSIMD(m, n, alpha, x, incX, y, incY, a, lda) {
		return
	}
	gerPortableSIMD(m, n, alpha, x, incX, y, incY, a, lda)
}
