// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || noasm || gccgo || safe

package f64

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestGemvTZeroBetaTail(t *testing.T) {
	for _, m := range []int{0, 1, 8} {
		const n = 9
		a := make([]float64, m*n)
		x := make([]float64, m)
		for i := range a {
			a[i] = 2
		}
		for i := range x {
			x[i] = 3
		}
		y := make([]float64, n+3)
		for i := range y[:n] {
			y[i] = math.NaN()
		}
		copy(y[n:], []float64{7, 8, 9})
		GemvT(uintptr(m), n, 0.5, a, n, x, 1, 0, y, 1)
		for i, got := range y {
			want := float64(3 * m)
			if i >= n {
				want = float64(i - n + 7)
			}
			if got != want {
				t.Errorf("m=%d: y[%d]=%g, want %g", m, i, got, want)
			}
		}
	}
}

func TestGemvTStridedOrder(t *testing.T) {
	for _, shape := range [][2]int{
		{0, 0}, {0, 10}, {1, 0}, {1, 1}, {3, 5}, {9, 8}, {9, 11},
		{9, 31}, {9, 32}, {9, 33},
		{1000, 10}, {1000, 12}, {1000, 31}, {1000, 32}, {1000, 33},
	} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, incs := range [][2]int{{2, 3}, {-2, 3}, {2, -3}, {-2, -3}, {1, 3}, {2, 1}, {0, 3}, {2, 0}} {
			incX, incY := incs[0], incs[1]
			for _, beta := range []float64{0, -0.5, 1} {
				t.Run(fmt.Sprintf("%dx%d/x=%d/y=%d/beta=%g", m, n, incX, incY, beta), func(t *testing.T) {
					a := gemvTTestData(m * lda)
					x := gemvTTestData(max(0, m-1)*max(incX, -incX) + 1)
					y := gemvTTestData(max(0, n-1)*max(incY, -incY) + 4)
					if m == 3 && n == 5 {
						copy(a, []float64{math.Inf(1), math.NaN(), math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.MaxFloat64})
					}
					want := slices.Clone(y)
					gemvTStridedReference(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, want, uintptr(incY))
					GemvT(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
					gemvTCheckBits(t, y, want)
				})
			}
		}
	}
}

func TestGemvTStridedOverlap(t *testing.T) {
	const m = 9
	for _, n := range []int{10, 31, 32, 33} {
		lda := n + 3
		for _, incs := range [][2]int{{2, 3}, {-2, 3}, {2, -3}, {-2, -3}, {2, 0}} {
			incX, incY := incs[0], incs[1]
			for _, beta := range []float64{0, 1, -0.5} {
				for _, matrixOverlap := range []bool{false, true} {
					t.Run(fmt.Sprintf("n=%d/x=%d/y=%d/beta=%g/matrix=%t", n, incX, incY, beta, matrixOverlap), func(t *testing.T) {
						got := gemvTTestData(m*lda + 32)
						want := slices.Clone(got)
						a := gemvTTestData(m * lda)
						wa := a
						if matrixOverlap {
							a, wa = got[2:], want[2:]
						}
						gemvTStridedReference(m, uintptr(n), -0.75, wa, uintptr(lda), want[3:], uintptr(incX), beta, want[5:], uintptr(incY))
						GemvT(m, uintptr(n), -0.75, a, uintptr(lda), got[3:], uintptr(incX), beta, got[5:], uintptr(incY))
						gemvTCheckBits(t, got, want)
					})
				}
			}
		}
	}
}

func gemvTTestData(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64((i*17)%31-15) / 7
	}
	return x
}

func gemvTCheckBits(t *testing.T, got, want []float64) {
	t.Helper()
	for i, v := range got {
		if math.Float64bits(v) != math.Float64bits(want[i]) && !(math.IsNaN(v) && math.IsNaN(want[i])) {
			t.Fatalf("index %d: got %g (%x), want %g (%x)", i, v, math.Float64bits(v), want[i], math.Float64bits(want[i]))
		}
	}
}

func gemvTStridedReference(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var kx, ky uintptr
	if int(incX) < 0 {
		kx = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		ky = uintptr(-int(n-1) * int(incY))
	}
	if beta == 0 {
		for j, iy := uintptr(0), ky; j < n; j++ {
			y[iy] = 0
			iy += incY
		}
	} else if int(incY) < 0 {
		ScalInc(beta, y, n, -incY)
	} else {
		ScalInc(beta, y, n, incY)
	}
	for i, ix := uintptr(0), kx; i < m; i++ {
		AxpyInc(alpha*x[ix], a[lda*i:lda*i+n], y, n, 1, incY, 0, ky)
		ix += incX
	}
}
