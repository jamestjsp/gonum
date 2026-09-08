// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestSdsdotWidenedBias(t *testing.T) {
	// Every product and every partial sum, including alpha, is exactly
	// representable in float64. Thus an alpha-inclusive scalar oracle does
	// not impose an addition order on the implementation.
	for _, n := range []int{2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129} {
		for _, incX := range []int{1, -1, 3, -3} {
			for _, incY := range []int{1, -1, 3, -3} {
				for _, sign := range []float32{1, -1} {
					t.Run(fmt.Sprintf("n=%d/incX=%d/incY=%d/sign=%g", n, incX, incY, sign), func(t *testing.T) {
						xValues, yValues := make([]float32, n), make([]float32, n)
						xValues[0], xValues[n-1] = -sign*0x1p24, sign*0.5
						for i := range yValues {
							yValues[i] = 1
						}
						x, y := sdsdotStridedValues(xValues, incX), sdsdotStridedValues(yValues, incY)
						alpha := sign * 0x1p24
						want := float64(alpha)
						for i, v := range xValues {
							want += float64(v) * float64(yValues[i])
						}
						sdsdotCheck(t, n, alpha, x, incX, y, incY, float32(want))
					})
				}
			}
		}
	}
}

func TestSdsdotIEEE(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := math.Float32frombits(0x7fc12345)
	negativeZero := math.Float32frombits(1 << 31)
	for _, test := range []struct {
		name  string
		alpha float32
		x, y  []float32
		want  float32
	}{
		{name: "EmptyBias", alpha: 3, want: 3},
		{name: "EmptyPositiveZero", alpha: 0, want: 0},
		{name: "EmptyNegativeZero", alpha: negativeZero, want: negativeZero},
		{name: "EmptyInfinity", alpha: inf, want: inf},
		{name: "EmptyNaN", alpha: nan, want: nan},
		{name: "AvoidIntermediateOverflow", alpha: -math.MaxFloat32, x: []float32{math.MaxFloat32}, y: []float32{2}, want: math.MaxFloat32},
		{name: "AvoidIntermediateUnderflow", alpha: math.SmallestNonzeroFloat32, x: []float32{math.SmallestNonzeroFloat32}, y: []float32{0.5}, want: 2 * math.SmallestNonzeroFloat32},
		{name: "FinalOverflow", alpha: math.MaxFloat32, x: []float32{math.MaxFloat32}, y: []float32{1}, want: inf},
		{name: "FinalSubnormal", alpha: 0, x: []float32{math.SmallestNonzeroFloat32}, y: []float32{1}, want: math.SmallestNonzeroFloat32},
		{name: "ExactCancellation", alpha: -1, x: []float32{1}, y: []float32{1}, want: 0},
		{name: "InfiniteBias", alpha: inf, x: []float32{1}, y: []float32{1}, want: inf},
		{name: "InfiniteProduct", alpha: 1, x: []float32{inf}, y: []float32{1}, want: inf},
		{name: "OpposingInfinities", alpha: -inf, x: []float32{inf}, y: []float32{1}, want: nan},
		{name: "NaNBias", alpha: nan, x: []float32{1}, y: []float32{1}, want: nan},
		{name: "NaNProduct", alpha: 1, x: []float32{nan}, y: []float32{1}, want: nan},
	} {
		for _, incX := range []int{1, -1, 3, -3} {
			for _, incY := range []int{1, -1, 3, -3} {
				t.Run(fmt.Sprintf("%s/incX=%d/incY=%d", test.name, incX, incY), func(t *testing.T) {
					x, y := sdsdotStridedValues(test.x, incX), sdsdotStridedValues(test.y, incY)
					sdsdotCheck(t, len(test.x), test.alpha, x, incX, y, incY, test.want)
				})
			}
		}
	}
}

func TestSdsdotReadOnlyAlias(t *testing.T) {
	x := []float32{1, -2, 3, -4, 5}
	sdsdotCheck(t, len(x), 0.5, x, 1, x, -1, 35.5)
}

func TestSdsdotValidationOrder(t *testing.T) {
	for _, test := range []struct {
		name       string
		n          int
		incX, incY int
		x, y       []float32
		want       string
	}{
		{name: "ZeroXBeforeYAndN", n: -1, incX: 0, incY: 0, want: zeroIncX},
		{name: "ZeroYBeforeN", n: -1, incX: 1, incY: 0, want: zeroIncY},
		{name: "EmptyStillChecksX", n: 0, incX: 0, incY: 1, want: zeroIncX},
		{name: "EmptyStillChecksY", n: 0, incX: 1, incY: 0, want: zeroIncY},
		{name: "NegativeN", n: -1, incX: 1, incY: 1, want: nLT0},
		{name: "ShortXBeforeY", n: 1, incX: 1, incY: 1, want: shortX},
		{name: "ShortY", n: 1, incX: 1, incY: 1, x: []float32{1}, want: shortY},
		{name: "StridedShortX", n: 2, incX: -3, incY: 3, x: []float32{1}, y: []float32{1}, want: shortX},
		{name: "StridedShortY", n: 2, incX: 3, incY: -3, x: []float32{1, 0, 0, 1}, y: []float32{1}, want: shortY},
	} {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if got := recover(); got != test.want {
					t.Errorf("unexpected panic: got %v, want %q", got, test.want)
				}
			}()
			(Implementation{}).Sdsdot(test.n, 3, test.x, test.incX, test.y, test.incY)
		})
	}
}

func sdsdotStridedValues(values []float32, inc int) []float32 {
	if len(values) == 0 {
		return nil
	}
	absInc := inc
	if absInc < 0 {
		absInc = -absInc
	}
	x := make([]float32, (len(values)-1)*absInc+1)
	for i := range x {
		x[i] = math.Float32frombits(0x7fc12345)
	}
	ix := 0
	if inc < 0 {
		ix = len(x) - 1
	}
	for _, v := range values {
		x[ix] = v
		ix += inc
	}
	return x
}

func sdsdotCheck(t *testing.T, n int, alpha float32, x []float32, incX int, y []float32, incY int, want float32) {
	t.Helper()
	originalX, originalY := slices.Clone(x), slices.Clone(y)
	got := (Implementation{}).Sdsdot(n, alpha, x, incX, y, incY)
	if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
		t.Errorf("got %g (%08x), want %g (%08x)", got, math.Float32bits(got), want, math.Float32bits(want))
	}
	equalBits := func(a, b float32) bool { return math.Float32bits(a) == math.Float32bits(b) }
	if !slices.EqualFunc(x, originalX, equalBits) || !slices.EqualFunc(y, originalY, equalBits) {
		t.Fatal("Sdsdot modified an input")
	}
}
