// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"slices"
	"testing"
)

func TestDivShortSIMDIEEEAndOverlap(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -3, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	check := func(got, want []float64) {
		t.Helper()
		for i, g := range got {
			w := want[i]
			if math.Float64bits(g) != math.Float64bits(w) && !(math.IsNaN(g) && math.IsNaN(w)) {
				t.Fatalf("index=%d got=%g (%x) want=%g (%x)", i, g, math.Float64bits(g), w, math.Float64bits(w))
			}
		}
	}
	for n := 0; n <= 65; n++ {
		for offset := 0; offset < len(values); offset++ {
			x, y := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = values[i%len(values)]
				y[i] = values[(i+offset)%len(values)]
			}
			want := make([]float64, n)
			for i := range x {
				want[i] = x[i] / y[i]
			}
			dst := append(make([]float64, n), 99)
			ret := DivToSIMD(dst, x, y)
			if len(ret) != n+1 || ret[n] != 99 {
				t.Fatal("DivTo changed destination length or suffix")
			}
			check(dst[:n], want)
			copy(dst, x)
			DivSIMD(dst, y)
			check(dst[:n], want)
			copy(dst, y)
			DivToSIMD(dst, x, dst)
			check(dst[:n], want)
		}
		for _, offset := range []int{-2, -1, 0, 1, 2} {
			data := make([]float64, n+4)
			for i := range data {
				data[i] = float64(i + 2)
			}
			want := slices.Clone(data)
			x, y := data[2:2+n], data[2+offset:2+offset+n]
			wx, wy := want[2:2+n], want[2+offset:2+offset+n]
			for i := range wx {
				wy[i] = wx[i] / wy[i]
			}
			DivToSIMD(y, x, y)
			check(data, want)
		}
	}
}
