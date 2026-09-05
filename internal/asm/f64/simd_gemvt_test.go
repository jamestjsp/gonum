// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"simd"
	"slices"
	"testing"
)

func TestSIMDMatrixGemvTRowRemainder(t *testing.T) {
	width := simd.BroadcastFloat64s(0).Len()
	for _, m := range []int{1, 2, 3, 5, 6, 7, 9, 15} {
		for _, n := range []int{4*width - 1, 4 * width, 4*width + 1, 8*width + 1} {
			for _, incX := range []int{1, -2} {
				for _, beta := range []float64{0, -0.5} {
					t.Run(fmt.Sprintf("%dx%d/incX=%d/beta=%g", m, n, incX, beta), func(t *testing.T) {
						lda := n + 3
						a, _ := matrixVector(m*lda, 1)
						x, ix := matrixVector(m, incX)
						y, _ := matrixVector(n+3, 1)
						if beta == 0 {
							for i := range y[:n] {
								y[i] = math.NaN()
							}
						}
						want := slices.Clone(y)
						for j := range want[:n] {
							if beta == 0 {
								want[j] = 0
							} else {
								want[j] *= beta
							}
						}
						for i := 0; i < m; i++ {
							scale := -0.75 * x[ix+i*incX]
							for j := 0; j < n; j++ {
								want[j] += scale * a[i*lda+j]
							}
						}
						GemvTSIMD(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, y, 1)
						portableSIMDCheckBits(t, "residual rows", y, want)
					})
				}
			}
		}
	}
}
