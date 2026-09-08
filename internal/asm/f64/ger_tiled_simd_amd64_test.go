// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"testing"
)

func TestGerTiledExact(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -3, 0.25, math.MaxFloat64, math.Inf(1), math.NaN()}
	for _, m := range []int{1, 3, 4, 5, 63, 64, 65} {
		for _, n := range []int{3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65} {
			for _, inc := range []int{1, 2, 3, 7} {
				for _, alpha := range values {
					lda := n + 3
					x, y := make([]float64, (m-1)*inc+1), make([]float64, (n-1)*inc+1)
					for i := range x {
						x[i] = values[i%len(values)]
					}
					for i := range y {
						y[i] = values[(i+3)%len(values)]
					}
					a := make([]float64, (m-1)*lda+n+2)
					for i := range a {
						a[i] = float64(i+1) * 0.125
					}
					want := append([]float64(nil), a...)
					for i := 0; i < m; i++ {
						scale := float64(alpha * x[i*inc])
						for j := 0; j < n; j++ {
							product := float64(scale * y[j*inc])
							want[i*lda+j] = product + want[i*lda+j]
						}
					}
					GerSIMD(uintptr(m), uintptr(n), alpha, x, uintptr(inc), y, uintptr(inc), a, uintptr(lda))
					for i, v := range a {
						if math.Float64bits(v) != math.Float64bits(want[i]) && !(math.IsNaN(v) && math.IsNaN(want[i])) {
							t.Fatalf("m=%d n=%d inc=%d alpha=%g index=%d got=%g want=%g", m, n, inc, alpha, i, v, want[i])
						}
					}
				}
			}
		}
	}
}

func TestGerTiledEligibility(t *testing.T) {
	x, y, a := make([]float64, 16), make([]float64, 16), make([]float64, 64)
	enabled := !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2()
	if got := gerTiledHardwareSIMD(4, 4, 0.5, x, 3, y, 3, a, 7); got != enabled {
		t.Fatalf("valid tile eligibility got=%t want=%t", got, enabled)
	}
	for _, tc := range []struct{ m, n, ix, iy, lda uintptr }{
		{0, 4, 1, 1, 4}, {4, 0, 1, 1, 4}, {3, 4, 1, 1, 4},
		{4, 4, 0, 1, 4}, {4, 4, 1, 0, 4}, {4, 4, ^uintptr(0), 1, 4},
		{4, 4, 1, ^uintptr(0), 4}, {4, 4, 1, 1, 3}, {4, 4, 1, 1, ^uintptr(0)},
		{^uintptr(0), 4, 1, 1, 4}, {4, ^uintptr(0), 1, 1, 4}, {4, 4, 16, 1, 4},
		{4, 65, 1, 1, 65}, {4, 4, 1, 1, 22},
	} {
		for i := range a {
			a[i] = 7
		}
		if gerTiledHardwareSIMD(tc.m, tc.n, 0.5, x, tc.ix, y, tc.iy, a, tc.lda) {
			t.Fatalf("accepted invalid tile %+v", tc)
		}
		for i, v := range a {
			if v != 7 {
				t.Fatalf("declined tile changed a[%d]", i)
			}
		}
	}
	for _, xAlias := range []bool{true, false} {
		if xAlias {
			if gerTiledHardwareSIMD(4, 4, 0.5, a[:4], 1, y, 1, a, 4) {
				t.Fatal("accepted x alias")
			}
		} else if gerTiledHardwareSIMD(4, 4, 0.5, x, 1, a[:4], 1, a, 4) {
			t.Fatal("accepted y alias")
		}
	}
}

func TestGerTiledAliasFallback(t *testing.T) {
	for _, aliasX := range []bool{false, true} {
		for _, n := range []int{7, 16, 33} {
			const m = 5
			lda := n + 3
			backing := make([]float64, m*lda+20)
			for i := range backing {
				backing[i] = float64(i%13-6) * 0.125
			}
			want := append([]float64(nil), backing...)
			other := make([]float64, n+m)
			for i := range other {
				other[i] = float64(i+1) * 0.125
			}
			x, y, wx, wy := other[:m], backing[1:1+n], other[:m], want[1:1+n]
			if aliasX {
				x, y, wx, wy = backing[1:1+m], other[:n], want[1:1+m], other[:n]
			}
			GerSIMD(m, uintptr(n), 0.5, x, 1, y, 1, backing, uintptr(lda))
			gerPortableSIMD(m, uintptr(n), 0.5, wx, 1, wy, 1, want, uintptr(lda))
			for i, v := range backing {
				if math.Float64bits(v) != math.Float64bits(want[i]) {
					t.Fatalf("aliasX=%t n=%d index=%d got=%g want=%g", aliasX, n, i, v, want[i])
				}
			}
		}
	}
}
