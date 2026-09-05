// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestSIMDDdotWidenBeforeMultiply(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for n := 0; n <= 129; n++ {
		x, y := make([]float32, n), make([]float32, n)
		var want, magnitude float64
		for i := range x {
			x[i] = float32(math.Ldexp(rng.Float64()-0.5, rng.IntN(220)-110))
			y[i] = float32(math.Ldexp(rng.Float64()-0.5, rng.IntN(220)-110))
			p := float64(x[i]) * float64(y[i])
			want += p
			magnitude += math.Abs(p)
		}
		check := func(got float64) {
			t.Helper()
			if math.IsNaN(got) || math.Abs(got-want) > 3e-15*magnitude {
				t.Fatalf("n=%d got %g want %g", n, got, want)
			}
		}
		check(DdotUnitarySIMD(x, y))
		for _, inc := range []int{1, 2, 3, 7} {
			xs, ys := make([]float32, n*inc), make([]float32, n*inc)
			for i := range xs {
				xs[i], ys[i] = float32(math.NaN()), float32(math.NaN())
			}
			for i := range x {
				xs[i*inc], ys[i*inc] = x[i], y[i]
			}
			check(DdotIncSIMD(xs, ys, uintptr(n), uintptr(inc), uintptr(inc), 0, 0))
			if n > 0 {
				check(DdotIncSIMD(xs, ys, uintptr(n), uintptr(-inc), uintptr(-inc), uintptr((n-1)*inc), uintptr((n-1)*inc)))
			}
		}
	}
}

func TestSIMDDdotSpecialValues(t *testing.T) {
	for _, n := range []int{1, 7, 8, 9, 16, 17, 33} {
		for _, special := range []float32{math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			for pos := 0; pos < n; pos++ {
				x, y := make([]float32, n), make([]float32, n)
				for i := range x {
					y[i] = 2
				}
				x[pos] = special
				want := float64(special) * 2
				check := func(got float64) {
					t.Helper()
					if got != want && !(math.IsNaN(got) && math.IsNaN(want)) {
						t.Fatalf("n=%d pos=%d special=%g: got %g want %g", n, pos, special, got, want)
					}
				}
				check(DdotUnitarySIMD(x, y))
				xs := make([]float32, 2*n)
				for i, v := range x {
					xs[2*i] = v
				}
				check(DdotIncSIMD(xs, y, uintptr(n), 2, 1, 0, 0))
				check(DdotIncSIMD(xs, []float32{2}, uintptr(n), 2, 0, 0, 0))
			}
		}
	}
}
