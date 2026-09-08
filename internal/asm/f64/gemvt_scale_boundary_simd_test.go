// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestGemvTUnitScaleBoundaries(t *testing.T) {
	for _, m := range []int{0, 1, 3, 4, 5, 8} {
		for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129} {
			for _, beta := range []float64{0, math.Copysign(0, -1), .25, -1, math.Inf(1), math.NaN()} {
				lda := n + 3
				x, a, y := make([]float64, m), make([]float64, max(0, (m-1)*lda+n)), make([]float64, n+3)
				for i := range x {
					x[i] = float64(i%5-2) * .25
				}
				for i := range a {
					a[i] = float64(i%11-5) * .0625
				}
				for i := range y {
					y[i] = []float64{0, math.Copysign(0, -1), 1, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.NaN()}[i%7]
				}
				want := append([]float64(nil), y...)
				for j := 0; j < n; j++ {
					if beta == 0 {
						want[j] = 0
					} else {
						want[j] = float64(beta * want[j])
					}
				}
				for i := 0; i < m; i++ {
					scale := float64(.5 * x[i])
					for j := 0; j < n; j++ {
						product := float64(scale * a[i*lda+j])
						want[j] += product
					}
				}
				GemvTSIMD(uintptr(m), uintptr(n), .5, a, uintptr(lda), x, 1, beta, y, 1)
				checkGemvTEightBits(t, y, want)
			}
		}
	}
}
