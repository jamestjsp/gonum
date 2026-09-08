// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
)

func gemvTSmallBackendReference(m, n, lda int, alpha float64, a, x []float64, beta float64, y []float64) {
	for j := 0; j < n; j++ {
		if beta == 0 {
			y[j] = 0
		} else {
			y[j] = float64(beta * y[j])
		}
	}
	width := simd.BroadcastFloat64s(0).Len()
	vectorEnd := n - n%width
	for i := 0; i < m; i++ {
		scale := float64(alpha * x[i])
		for j := 0; j < vectorEnd; j++ {
			product := float64(scale * a[i*lda+j])
			y[j] += product
		}
		for j := vectorEnd; j < n; j++ {
			y[j] = math.FMA(scale, a[i*lda+j], y[j])
		}
	}
}
