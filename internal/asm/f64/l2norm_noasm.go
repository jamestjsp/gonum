// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && (!arm64 || !go1.27 || !goexperiment.simd)) || noasm || gccgo || safe

package f64

// L2NormUnitary returns the L2-norm of x.
func L2NormUnitary(x []float64) float64 {
	return l2NormUnitaryScalar(x)
}

// L2NormInc returns the L2-norm of x.
func L2NormInc(x []float64, n, incX uintptr) float64 {
	if incX == 0 {
		return 0
	}
	return l2NormIncScalar(x, n, incX)
}

// L2DistanceUnitary returns the L2-norm of x-y.
func L2DistanceUnitary(x, y []float64) float64 {
	return l2DistanceUnitaryScalar(x, y)
}
