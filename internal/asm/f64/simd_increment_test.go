// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"slices"
	"testing"
)

func TestSIMDIncrementDependencies(t *testing.T) {
	for _, n := range []uintptr{1, 7, 8, 9, 16, 17, 33} {
		for _, inc := range []uintptr{0, 1, 2} {
			for _, outInc := range []uintptr{0, 1, 2} {
				for _, offset := range []uintptr{0, 1} {
					x := make([]float64, 2*n+2)
					for i := range x {
						x[i] = float64(i+1) / 16
					}
					want := slices.Clone(x)
					ix, iy := uintptr(0), offset
					for i := uintptr(0); i < n; i++ {
						want[iy] += .5 * want[ix]
						ix += inc
						iy += outInc
					}
					AxpyIncSIMD(.5, x, x, n, inc, outInc, 0, offset)
					if !slices.Equal(x, want) {
						t.Fatalf("Axpy overlap n=%d inc=%d,%d offset=%d", n, inc, outInc, offset)
					}
					for i := range x {
						x[i] = float64(i+1) / 16
					}
					want = slices.Clone(x)
					ix, iy = 0, 0
					for i := uintptr(0); i < n; i++ {
						want[iy] = .5 * want[ix]
						ix += inc
						iy += outInc
					}
					ScalIncToSIMD(x, outInc, .5, x, n, inc)
					if !slices.Equal(x, want) {
						t.Fatalf("ScalIncTo overlap n=%d inc=%d,%d", n, inc, outInc)
					}
				}
			}
		}
		x := []float64{-1.125}
		want := x[0]
		for i := uintptr(0); i < n; i++ {
			want *= -.5
		}
		ScalIncSIMD(-.5, x, n, 0)
		if x[0] != want {
			t.Fatalf("ScalInc zero n=%d got %g want %g", n, x[0], want)
		}
	}
}
