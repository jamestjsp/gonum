// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestGemvNBlockBoundariesSIMD(t *testing.T) {
	for _, m := range []int{0, 1, 3, 4, 5, 8, 12, 65} {
		for n := 0; n <= 65; n++ {
			for _, beta := range []float64{0, .75} {
				lda := n + 3
				a, x, y := make([]float64, m*lda), make([]float64, n), make([]float64, m)
				for i := range a {
					a[i] = -99
				}
				for j := range x {
					x[j] = float64(j%9-4) * .125
				}
				want := make([]float64, m)
				for i := range y {
					y[i] = float64(i%7-3) * .5
					for j := 0; j < n; j++ {
						a[i*lda+j] = float64((i+j)%11-5) * .0625
						want[i] += .5 * a[i*lda+j] * x[j]
					}
					want[i] += beta * y[i]
					if beta == 0 {
						y[i] = math.NaN()
					}
				}
				GemvNSIMD(uintptr(m), uintptr(n), .5, a, uintptr(lda), x, 1, beta, y, 1)
				for i, got := range y {
					if got != want[i] {
						t.Fatalf("m=%d n=%d beta=%g row=%d got=%g want=%g", m, n, beta, i, got, want[i])
					}
				}
			}
		}
	}
}

func TestGemvNBlockGroupingRecoverySIMD(t *testing.T) {
	const m, n, lda = 12, 16, 19
	a, x, y := make([]float64, (m-1)*lda+n), make([]float64, n), make([]float64, m)
	for i := range x {
		x[i] = 1
	}
	for i := range y {
		y[i] = 1
		a[i*lda] = 2
		if i >= 4 {
			a[i*lda] = math.MaxFloat64
			a[i*lda+1] = -math.MaxFloat64
			a[i*lda+4] = math.MaxFloat64
		}
	}
	want := append([]float64(nil), y...)
	gemvNPortableSIMD(m, n, .5, a, lda, x, 1, .25, want, 1)
	GemvNSIMD(m, n, .5, a, lda, x, 1, .25, y, 1)
	for i, got := range y {
		if got != want[i] && !(math.IsNaN(got) && math.IsNaN(want[i])) {
			t.Fatalf("row=%d got=%g want=%g", i, got, want[i])
		}
	}
}
