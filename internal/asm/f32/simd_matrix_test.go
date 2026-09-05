// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// matrixVector includes increment gaps, so every comparison also verifies that
// untouched entries and matrix padding are preserved.
func matrixVector(n, inc int) ([]float32, int) {
	size, first := 1, 0
	if n > 0 {
		size += (n - 1) * matrixAbs(inc)
	}
	if inc < 0 {
		first = size - 1
	}
	v := make([]float32, size)
	for i := range v {
		v[i] = float32(i%19-9) * 0.125
	}
	return v, first
}

func matrixCheck(t *testing.T, got, want []float32) {
	t.Helper()
	for i, v := range want {
		g := got[i]
		if g == v || math.IsNaN(float64(g)) && math.IsNaN(float64(v)) {
			continue
		}
		if math.Abs(float64(g-v)) <= 3e-5*(1+math.Abs(float64(v))) {
			continue
		}
		t.Fatalf("element %d: got %.18g want %.18g", i, g, v)
	}
}

func TestSIMDMatrixGer(t *testing.T) {
	for _, shape := range [][2]int{{0, 0}, {0, 9}, {1, 0}, {1, 1}, {2, 7}, {3, 9}, {4, 16}, {5, 17}, {7, 31}, {8, 32}, {9, 33}, {16, 65}, {65, 7}, {64, 64}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, incX := range []int{-2, -1, 0, 1, 2} {
			for _, incY := range []int{-2, -1, 0, 1, 2} {
				t.Run(fmt.Sprintf("%dx%d/x=%d/y=%d", m, n, incX, incY), func(t *testing.T) {
					x, ix := matrixVector(m, incX)
					y, iy := matrixVector(n, incY)
					a, _ := matrixVector(m*lda, 1)
					want := slices.Clone(a)
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							want[i*lda+j] += (-0.75 * x[ix+i*incX]) * y[iy+j*incY]
						}
					}
					GerSIMD(uintptr(m), uintptr(n), -0.75, x, uintptr(incX), y, uintptr(incY), a, uintptr(lda))
					matrixCheck(t, a, want)
				})
			}
		}
	}
}

func TestSIMDMatrixGerOverlap(t *testing.T) {
	for _, offset := range []int{0, 1, 7} {
		const m, n, lda = 8, 17, 19
		a, _ := matrixVector(m*lda, 1)
		want := slices.Clone(a)
		x, _ := matrixVector(m, 1)
		y := a[offset : offset+n]
		wy := want[offset : offset+n]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				want[i*lda+j] += (-0.75 * x[i]) * wy[j]
			}
		}
		GerSIMD(m, n, -0.75, x, 1, y, 1, a, lda)
		matrixCheck(t, a, want)
	}
}

func matrixAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
