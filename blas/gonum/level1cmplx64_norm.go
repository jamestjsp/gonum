// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

func scnrm2Unitary(x []complex64) (float32, bool) {
	return scnrm2Inc(len(x), x, 1)
}

func scnrm2Inc(n int, x []complex64, incX int) (float32, bool) {
	// Every finite float32 square fits exactly in float64, including subnormals.
	var s0, s1, s2, s3 float64
	i, ix := 0, 0
	for ; i+2 <= n; i += 2 {
		v0, v1 := x[ix], x[ix+incX]
		r0, i0 := float64(real(v0)), float64(imag(v0))
		r1, i1 := float64(real(v1)), float64(imag(v1))
		s0 += r0 * r0
		s1 += i0 * i0
		s2 += r1 * r1
		s3 += i1 * i1
		ix += 2 * incX
	}
	if i < n {
		r, im := float64(real(x[ix])), float64(imag(x[ix]))
		s0 += r * r
		s1 += im * im
	}
	sum := (s0 + s1) + (s2 + s3)
	// Retry the scaled recurrence to retain complex norm's Inf-over-NaN rule.
	return float32(math.Sqrt(sum)), !math.IsNaN(sum)
}
