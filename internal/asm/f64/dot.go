// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 && (!arm64 || !go1.27 || !goexperiment.simd)) || noasm || gccgo || safe

package f64

// DotUnitary is
//
//	for i, v := range x {
//		sum += y[i] * v
//	}
//	return sum
func DotUnitary(x, y []float64) (sum float64) {
	for i, v := range x {
		sum += y[i] * v
	}
	return sum
}
