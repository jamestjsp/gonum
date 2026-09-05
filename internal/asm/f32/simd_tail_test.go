// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"math/rand/v2"
	"simd"
	"slices"
	"sync"
	"testing"
)

func TestSIMDTailReductionAccuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	for n := 0; n <= 257; n++ {
		x, y := make([]float32, n), make([]float32, n)
		var sum, dot, sumMagnitude, dotMagnitude float64
		for i := range x {
			x[i] = float32(math.Ldexp(rng.Float64()-0.5, rng.IntN(80)-40))
			y[i] = float32(math.Ldexp(rng.Float64()-0.5, rng.IntN(80)-40))
			p := x[i] * y[i]
			sum += float64(x[i])
			sumMagnitude += math.Abs(float64(x[i]))
			dot += float64(p)
			dotMagnitude += math.Abs(float64(p))
		}
		check := func(name string, got float32, want, magnitude float64) {
			t.Helper()
			if math.IsNaN(float64(got)) || math.Abs(float64(got)-want) > 2e-6*magnitude {
				t.Fatalf("%s n=%d: got%g want%g magnitude%g", name, n, got, want, magnitude)
			}
		}
		check("Sum", SumSIMD(x), sum, sumMagnitude)
		check("Dot", DotUnitarySIMD(x, y), dot, dotMagnitude)
		for _, inc := range []int{2, 3, 7} {
			xs, ys := make([]float32, max(0, (n-1)*inc+1)), make([]float32, max(0, (n-1)*inc+1))
			for i := range xs {
				xs[i], ys[i] = float32(math.NaN()), float32(math.NaN())
			}
			for i := range x {
				xs[i*inc], ys[i*inc] = x[i], y[i]
			}
			check("DotInc", DotIncSIMD(xs, ys, uintptr(n), uintptr(inc), uintptr(inc), 0, 0), dot, dotMagnitude)
			if n > 0 {
				check("DotIncReverse", DotIncSIMD(xs, ys, uintptr(n), uintptr(-inc), uintptr(-inc), uintptr((n-1)*inc), uintptr((n-1)*inc)), dot, dotMagnitude)
			}
		}
	}
}

func TestSIMDExactCapacityTails(t *testing.T) {
	specials := []float32{0, math.Float32frombits(1 << 31), math.SmallestNonzeroFloat32, float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 1, -2}
	for n := 0; n <= 65; n++ {
		x, y := make([]float32, n), make([]float32, n)
		for i := range x {
			x[i] = specials[i%len(specials)]
			y[i] = specials[(i+2)%len(specials)]
		}
		want := slices.Clone(y)
		for i, v := range x {
			want[i] = 0.5*v + y[i]
		}
		dst := make([]float32, n)
		AxpyUnitaryToSIMD(dst, 0.5, x, y)
		AxpyUnitarySIMD(0.5, x, y)
		for i, v := range want {
			for _, got := range []float32{y[i], dst[i]} {
				if math.Float32bits(got) != math.Float32bits(v) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(v))) {
					t.Fatalf("n=%d i=%d got%g want%g", n, i, got, v)
				}
			}
		}
	}
}

// Strided kernels may be applied concurrently to independent interleaved
// vectors. A vector load or read-modify-write that includes gaps would race.
func TestSIMDIndependentInterleavedVectors(t *testing.T) {
	const n, repeats = 65, 100
	x, y, dst := make([]float32, 2*n), make([]float32, 2*n), make([]float32, 2*n)
	for i := range x {
		x[i] = 1
	}
	var wg sync.WaitGroup
	for offset := 0; offset < 2; offset++ {
		wg.Go(func() {
			for k := 0; k < repeats; k++ {
				AxpyIncSIMD(0.5, x, y, n, 2, 2, uintptr(offset), uintptr(offset))
			}
		})
	}
	wg.Wait()
	for i, v := range y {
		if v != repeats*0.5 {
			t.Fatalf("AxpyInc i=%d got%g", i, v)
		}
	}
	for offset := 0; offset < 2; offset++ {
		wg.Go(func() {
			for k := 0; k < repeats; k++ {
				AxpyIncToSIMD(dst, 2, uintptr(offset), 0.5, x, y, n, 2, 2, uintptr(offset), uintptr(offset))
			}
		})
	}
	wg.Wait()
	for i, v := range dst {
		if v != (repeats+1)*0.5 {
			t.Fatalf("AxpyIncTo i=%d got%g", i, v)
		}
	}
}

func TestSIMDReductionFiniteCancellation(t *testing.T) {
	if !hardwareStridedSIMD || simd.Emulated() {
		t.Skip("regression for native AMD64 reduction regrouping")
	}
	m := float32(0.75 * math.MaxFloat32)
	for _, n := range []int{8, 16, 31, 32, 33, 64, 65} {
		x, y := make([]float32, n), make([]float32, n)
		for i := range y {
			y[i] = 1
		}
		x[0], x[1], x[4], x[5] = m, -m, m, -m
		if got := SumSIMD(x); got != 0 {
			t.Fatalf("Sum n=%d got%g want0", n, got)
		}
		if got := DotUnitarySIMD(x, y); got != 0 {
			t.Fatalf("Dot n=%d got%g want0", n, got)
		}
		xs, ys := make([]float32, 2*n), make([]float32, 2*n)
		for i := range x {
			xs[2*i], ys[2*i] = x[i], y[i]
		}
		if got := DotIncSIMD(xs, ys, uintptr(n), 2, 2, 0, 0); got != 0 {
			t.Fatalf("DotInc n=%d got%g want0", n, got)
		}
	}
}

// These arrangements cancel in the original 16-lane grouping, but both a
// narrower grouping and a whole-input sequential retry can overflow.
func TestSIMDReductionPreservesOriginalGrouping(t *testing.T) {
	if !hardwareStridedSIMD || simd.Emulated() || simd.VectorBitSize() != 512 {
		t.Skip("regression for original 16-lane AMD64 grouping")
	}
	m := float32(0.75 * math.MaxFloat32)
	t.Run("DotInc", func(t *testing.T) {
		const n = 32
		x, y := make([]float32, 2*n), make([]float32, 2*n)
		for i := 0; i < n; i++ {
			y[2*i] = 1
		}
		x[0], x[16], x[32], x[48] = m, m, -m, -m
		if got := DotIncSIMD(x, y, n, 2, 2, 0, 0); got != 0 {
			t.Fatalf("got%g want0 from original lane grouping", got)
		}
	})
	t.Run("FiniteTreeThenOverflowingTail", func(t *testing.T) {
		x, y := make([]float32, 17), make([]float32, 17)
		for i := range y {
			y[i] = 1
		}
		halfULP := float32(math.Ldexp(1, 103))
		x[0], x[1], x[3], x[16] = math.MaxFloat32, -halfULP, halfULP, halfULP
		want := math.Float32frombits(math.Float32bits(math.MaxFloat32) - 1)
		if got := SumSIMD(x); got != want {
			t.Errorf("Sum got%g want%g", got, want)
		}
		if got := DotUnitarySIMD(x, y); got != want {
			t.Errorf("Dot got%g want%g", got, want)
		}
	})
	t.Run("Horizontal", func(t *testing.T) {
		const n = 32
		x, y := make([]float32, n), make([]float32, n)
		for i := range y {
			y[i] = 1
		}
		x[0], x[8], x[20], x[28] = m, m, -m, -m
		if got := SumSIMD(x); got != 0 {
			t.Errorf("Sum got%g want0 from original lane grouping", got)
		}
		if got := DotUnitarySIMD(x, y); got != 0 {
			t.Errorf("Dot got%g want0 from original lane grouping", got)
		}
	})
}
