// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !safe && !noasm && !gccgo

package f64

import "testing"

// The BLAS API rejects zero increments. Preserve existing internal ASM behavior
// while repairing valid negative increments.
func TestGerASMZeroIncrements(t *testing.T) {
	x, y := []float64{2, 3, 4}, []float64{4, 6, 8}
	a := make([]float64, 9)
	Ger(3, 3, .5, x, 0, y, 1, a, 3)
	for i, v := range a {
		if v != y[i%3] {
			t.Fatalf("zero X index%d got%g want%g", i, v, y[i%3])
		}
	}
	a = make([]float64, 9)
	Ger(3, 3, .5, nil, 0, nil, 0, a, 3)
	for i, v := range a {
		if v != 0 {
			t.Fatalf("zero Y index%d changed", i)
		}
	}
}
