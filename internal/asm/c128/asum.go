// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || !go1.27 || !goexperiment.simd || noasm || gccgo || safe

package c128

import "math"

// AsumUnitary returns the sum of the absolute real and imaginary components.
func AsumUnitary(x []complex128) float64 {
	var sum float64
	for _, v := range x {
		sum += math.Abs(real(v)) + math.Abs(imag(v))
	}
	return sum
}
