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

func TestAxpyShortSIMDIEEEAndOverlap(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -3, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	for n := 0; n <= 65; n++ {
		for _, alpha := range []float64{0, math.Copysign(0, -1), .5, 2, math.Inf(1), math.NaN()} {
			x, y := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = values[i%len(values)]
				y[i] = values[(i+2)%len(values)]
			}
			want := make([]float64, n)
			for i := range x {
				want[i] = float64(alpha*x[i]) + y[i]
			}
			dst := make([]float64, n)
			AxpyUnitaryToSIMD(dst, alpha, x, y)
			portableSIMDCheckBits(t, "AxpyTo", dst, want)
			AxpyUnitarySIMD(alpha, x, y)
			portableSIMDCheckBits(t, "Axpy", y, want)
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
				wy[i] = .5*wx[i] + wy[i]
			}
			AxpyUnitaryToSIMD(y, .5, x, y)
			portableSIMDCheckBits(t, "AxpyTo overlap", data, want)
		}
	}
}
