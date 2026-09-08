// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"simd/archsimd"
	"testing"
)

type simdZdotHandoffShape struct {
	n, incX, incY int
}

type simdZdotHandoffFixture struct {
	name  string
	run   func()
	check func(testing.TB)
}

func TestSIMDZdotHandoffConsumerFixtures(t *testing.T) {
	fixtures := simdZdotHandoffFixtures()
	if len(fixtures) != 68 {
		t.Fatalf("fixture count: got %d want 68", len(fixtures))
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			for range 3 {
				fixture.run()
				fixture.check(t)
			}
		})
	}
}

func BenchmarkSIMDZdotHandoffConsumers(b *testing.B) {
	for _, fixture := range simdZdotHandoffFixtures() {
		b.Run(fixture.name, func(b *testing.B) {
			fixture.run()
			fixture.check(b)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fixture.run()
			}
			b.StopTimer()
			fixture.check(b)
		})
	}
}

func simdZdotHandoffClearAVX() {
	if archsimd.X86.AVX() {
		archsimd.ClearAVXUpperBits()
	}
}

func simdZdotHandoffFixtures() []simdZdotHandoffFixture {
	shapes := make([]simdZdotHandoffShape, 0, 34)
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 4096} {
		shapes = append(shapes, simdZdotHandoffShape{n: n, incX: 3, incY: 5})
	}
	for _, n := range []int{31, 4096} {
		for _, inc := range [][2]int{{1, 1}, {2, 2}, {3, 3}, {7, 7}, {16, 16}, {63, 63}, {-3, -5}, {3, -5}} {
			shapes = append(shapes, simdZdotHandoffShape{n: n, incX: inc[0], incY: inc[1]})
		}
	}
	fixtures := make([]simdZdotHandoffFixture, 0, 68)
	for _, shape := range shapes {
		fixtures = append(fixtures, simdZdotHandoffMakeFixture(shape, false))
		fixtures = append(fixtures, simdZdotHandoffMakeFixture(shape, true))
	}
	return fixtures
}

func simdZdotHandoffMakeFixture(shape simdZdotHandoffShape, conjugate bool) simdZdotHandoffFixture {
	n, incX, incY := shape.n, shape.incX, shape.incY
	x, ix := simdZdotHandoffVector(n, incX)
	y, iy := simdZdotHandoffVector(n, incY)
	// Build an independent component oracle in integers. X components have
	// denominator 16, Y components denominator 32, so every product has
	// denominator 512. At n<=4096 every intermediate integer numerator is
	// much smaller than 2^53. All float64 products and sums are therefore exact,
	// including the different SIMD and ASM accumulation groupings.
	var realSum, imagSum int64
	for i := 0; i < n; i++ {
		xr, xi := int64(i%17-8), int64((3*i+1)%19-9)
		yr, yi := int64((5*i+2)%23-11), int64((7*i+3)%29-14)
		x[ix] = complex(float64(xr)/16, float64(xi)/16)
		y[iy] = complex(float64(yr)/32, float64(yi)/32)
		if conjugate {
			xi = -xi
		}
		realSum += xr*yr - xi*yi
		imagSum += xr*yi + xi*yr
		ix += incX
		iy += incY
	}
	want := complex(float64(realSum)/512, float64(imagSum)/512)
	originalX := append([]complex128(nil), x...)
	originalY := append([]complex128(nil), y...)
	impl := Implementation{}
	var result complex128
	name := "Zdotu"
	// Select the method once during setup. There is no timed conjugation
	// branch, reset, phase counter or function-value call to the internal leaf.
	run := func() {
		simdZdotHandoffClearAVX()
		result = impl.Zdotu(n, x, incX, y, incY)
	}
	if conjugate {
		name = "Zdotc"
		run = func() {
			simdZdotHandoffClearAVX()
			result = impl.Zdotc(n, x, incX, y, incY)
		}
	}
	return simdZdotHandoffFixture{
		name: fmt.Sprintf("%s/n=%d/incX=%d/incY=%d", name, n, incX, incY),
		run:  run,
		check: func(t testing.TB) {
			t.Helper()
			if result != want {
				t.Fatalf("got %v want exact integer component sum %v", result, want)
			}
			if !simdZdotHandoffSameBits(x, originalX) || !simdZdotHandoffSameBits(y, originalY) {
				t.Fatal("dot modified an active element or stride gap")
			}
		},
	}
}

func simdZdotHandoffVector(n, inc int) (v []complex128, start int) {
	if n == 0 {
		return nil, 0
	}
	stride := inc
	if stride < 0 {
		stride = -stride
		start = (n - 1) * stride
	}
	// Exact highest active element: no extra capacity for an accidental end
	// access. Interior gaps are finite sentinels and checked bit-for-bit.
	v = make([]complex128, (n-1)*stride+1)
	for i := range v {
		v[i] = complex(-123.5, 789.25)
	}
	return v, start
}

func simdZdotHandoffSameBits(x, y []complex128) bool {
	if len(x) != len(y) {
		return false
	}
	for i, xv := range x {
		yv := y[i]
		if math.Float64bits(real(xv)) != math.Float64bits(real(yv)) ||
			math.Float64bits(imag(xv)) != math.Float64bits(imag(yv)) {
			return false
		}
	}
	return true
}
