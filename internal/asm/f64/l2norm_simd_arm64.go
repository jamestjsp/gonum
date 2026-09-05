// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import "math"

// L2NormUnitary returns the L2-norm of x.
func L2NormUnitary(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	if len(x) == 1 {
		return math.Abs(x[0])
	}
	if len(x) < 32 || l2NormPreferScalar(x[0]) {
		return l2NormUnitaryScalar(x)
	}
	return L2NormUnitarySIMD(x)
}

// L2NormInc returns the L2-norm of x.
func L2NormInc(x []float64, n, incX uintptr) float64 {
	if n == 0 || incX == 0 {
		return 0
	}
	if n == 1 {
		return math.Abs(x[0])
	}
	if n < 32 || l2NormPreferScalar(x[0]) {
		return l2NormIncScalar(x, n, incX)
	}
	return L2NormIncSIMD(x, n, incX)
}

// L2DistanceUnitary returns the L2-norm of x-y.
func L2DistanceUnitary(x, y []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	if len(x) == 1 {
		return math.Abs(x[0] - y[0])
	}
	if len(x) < 32 || l2NormPreferScalar(x[0]-y[0]) {
		return l2DistanceUnitaryScalar(x, y)
	}
	return L2DistanceUnitarySIMD(x, y)
}

// A first-value check avoids the speculative vector pass for zero-leading and
// commonly scaled inputs. Later exceptions are still handled by the SIMD retry.
func l2NormPreferScalar(v float64) bool {
	abs := math.Abs(v)
	return abs < 0x1p-485 || abs >= 0x1p512
}
