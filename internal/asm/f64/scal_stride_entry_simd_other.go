// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func ScalIncSIMD(alpha float64, x []float64, n, incX uintptr) { scalIncOriginalSIMD(alpha, x, n, incX) }
func ScalIncToSIMD(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr) {
	scalIncToOriginalSIMD(dst, incDst, alpha, x, n, incX)
}
