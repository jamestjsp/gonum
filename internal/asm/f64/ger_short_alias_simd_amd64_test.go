// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"slices"
	"testing"
)

func TestGerEightAliasedInputs(t *testing.T) {
	for _, m := range []int{4, 5} {
		for _, lda := range []int{8, 11} {
			for _, which := range []string{"x", "y"} {
				a := make([]float64, (m-1)*lda+8)
				for i := range a {
					a[i] = float64(i%11+1) * 0.25
				}
				w := slices.Clone(a)
				x, y := make([]float64, m), make([]float64, 8)
				for i := range x {
					x[i] = float64(i+1) * 0.5
				}
				for i := range y {
					y[i] = float64(i+1) * 0.125
				}
				wx, wy := slices.Clone(x), slices.Clone(y)
				if which == "x" {
					x = a[1 : 1+m]
					wx = w[1 : 1+m]
				} else {
					y = a[1:9]
					wy = w[1:9]
				}
				for i := 0; i < m; i++ {
					scale := float64(0.125 * wx[i])
					for j := 0; j < 8; j++ {
						w[i*lda+j] += float64(scale * wy[j])
					}
				}
				GerSIMD(uintptr(m), 8, 0.125, x, 1, y, 1, a, uintptr(lda))
				portableSIMDCheckBits(t, "Ger alias "+which, a, w)
			}
		}
	}
}
