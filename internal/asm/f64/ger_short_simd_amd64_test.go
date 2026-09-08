// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestGerEightExact(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -3, 0.25, math.MaxFloat64, math.Inf(1), math.NaN()}
	for m := 1; m <= 65; m++ {
		for _, lda := range []int{8, 11} {
			for _, alpha := range values {
				x := make([]float64, m)
				y := make([]float64, 8)
				a := make([]float64, (m-1)*lda+8)
				for i := range x {
					x[i] = values[i%len(values)]
				}
				for i := range y {
					y[i] = values[(i+3)%len(values)]
				}
				for i := range a {
					a[i] = float64(i+1) * 0.125
				}
				want := append([]float64(nil), a...)
				for i := 0; i < m; i++ {
					for j := 0; j < 8; j++ {
						scale := float64(alpha * x[i])
						prod := float64(scale * y[j])
						want[i*lda+j] = prod + want[i*lda+j]
					}
				}
				GerSIMD(uintptr(m), 8, alpha, x, 1, y, 1, a, uintptr(lda))
				for i, v := range a {
					if math.Float64bits(v) != math.Float64bits(want[i]) && !(math.IsNaN(v) && math.IsNaN(want[i])) {
						t.Fatalf("m=%d lda=%d alpha=%g index=%d got=%g want=%g", m, lda, alpha, i, v, want[i])
					}
				}
			}
		}
	}
}
