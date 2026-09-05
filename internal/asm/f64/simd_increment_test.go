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

func TestSIMDStridedDotOverflow(t *testing.T) {
	// Four accumulators overflow here before cancellation. Eight independent
	// lanes, used by the original 512-bit path, instead reduce to finite zero.
	m := 0.75 * math.MaxFloat64
	values := []float64{m, -m, 0, 0, m, -m, 0, 0}
	for _, inc := range []int{2, -2} {
		x, first := matrixVector(len(values), inc)
		y := slices.Clone(x)
		for i, v := range values {
			x[first+i*inc], y[first+i*inc] = v, 1
		}
		want := dotIncPortableSIMD(x, y, uintptr(len(values)), uintptr(inc), uintptr(inc), uintptr(first), uintptr(first))
		if simd.VectorBitSize() == 512 && want != 0 {
			t.Fatalf("expected finite zero from original grouping, got %g", want)
		}
		got := DotIncSIMD(x, y, uintptr(len(values)), uintptr(inc), uintptr(inc), uintptr(first), uintptr(first))
		if got != want && !(math.IsNaN(got) && math.IsNaN(want)) {
			t.Fatalf("inc=%d: got %g want %g", inc, got, want)
		}
	}
}

func TestSIMDStridedUnrollBoundaries(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 65, 129, 257} {
		for _, incs := range [][3]int{{2, 3, 7}, {-2, 3, -7}, {3, -7, 2}, {7, 16, 63}, {0, 3, 7}, {7, 0, 2}, {3, 7, 0}} {
			t.Run(fmt.Sprintf("n=%d/inc=%v", n, incs), func(t *testing.T) {
				incX, incY, incDst := incs[0], incs[1], incs[2]
				x, ix := matrixVector(n, incX)
				y, iy := matrixVector(n, incY)
				dst, idst := matrixVector(n, incDst)
				wantY, wantDst := slices.Clone(y), slices.Clone(dst)
				var dot float64
				for i := range n {
					xv, yv := x[ix+i*incX], y[iy+i*incY]
					dot += xv * yv
					wantY[iy+i*incY] += 0.5 * xv
					wantDst[idst+i*incDst] = 0.5*xv + yv
				}
				if got := DotIncSIMD(x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy)); got != dot {
					t.Fatalf("DotInc: got %g want %g", got, dot)
				}
				AxpyIncToSIMD(dst, uintptr(incDst), uintptr(idst), 0.5, x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy))
				if !slices.Equal(dst, wantDst) {
					t.Fatalf("AxpyIncTo: got %v want %v", dst, wantDst)
				}
				AxpyIncSIMD(0.5, x, y, uintptr(n), uintptr(incX), uintptr(incY), uintptr(ix), uintptr(iy))
				if !slices.Equal(y, wantY) {
					t.Fatalf("AxpyInc: got %v want %v", y, wantY)
				}
			})
		}
	}
}

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
