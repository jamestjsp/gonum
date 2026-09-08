// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math/big"
	"slices"
	"testing"
)

func TestCheckedDotStart(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	for _, n := range []int{1, 2, 3, 4, 9, 1 << 16, maxInt/4 + 1, maxInt/2 + 1, maxInt/2 + 2, maxInt} {
		for _, inc := range []int{minInt, minInt + 1, -maxInt / 2, -1 << 16, -4, -3, -1, 1, 3, 4, 1 << 16, maxInt / 2, maxInt} {
			for _, length := range []int{0, 1, 2, 3, 7, 16, 1 << 16, maxInt / 2, maxInt} {
				// The oracle uses mathematical integers, independent of the
				// machine-width product and unsigned MinInt handling.
				stride := big.NewInt(int64(inc))
				stride.Abs(stride)
				span := new(big.Int).Mul(big.NewInt(int64(n-1)), stride)
				wantOK := span.Cmp(big.NewInt(int64(length))) < 0
				var wantStart int
				if wantOK && inc < 0 {
					wantStart = int(span.Int64())
				}
				gotStart, gotOK := checkedDotStart(n, inc, length)
				if gotOK != wantOK || gotStart != wantStart {
					t.Errorf("n=%d inc=%d length=%d: got (%d,%v), want (%d,%v)", n, inc, length, gotStart, gotOK, wantStart, wantOK)
				}
			}
		}
	}
}

func TestDotSpanBoundedCalls(t *testing.T) {
	for _, incX := range []int{1, -1, 3, -3} {
		for _, incY := range []int{1, -1, 3, -3} {
			x := dotSpanStrided([]float64{1, -2, 3}, incX)
			y := dotSpanStrided([]float64{4, 5, -6}, incY)
			methods, unchanged := dotSpanTestMethods(x, y)
			for _, method := range methods {
				t.Run(fmt.Sprintf("%s/incX=%d/incY=%d", method.name, incX, incY), func(t *testing.T) {
					if got := method.run(3, incX, incY); got != -24+method.bias {
						t.Errorf("got %v, want %v", got, -24+method.bias)
					}
					if !unchanged() {
						t.Fatal("dot modified an input or a stride gap")
					}
				})
			}
		}
	}
}

func TestDotSpanSingleElementExtremeIncrements(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	for _, incX := range []int{minInt, -maxInt, -1, 1, maxInt} {
		for _, incY := range []int{minInt, -maxInt, -1, 1, maxInt} {
			methods, unchanged := dotSpanTestMethods([]float64{2}, []float64{3})
			for _, method := range methods {
				t.Run(fmt.Sprintf("%s/incX=%d/incY=%d", method.name, incX, incY), func(t *testing.T) {
					if got := method.run(1, incX, incY); got != 6+method.bias {
						t.Errorf("got %v, want %v", got, 6+method.bias)
					}
					if !unchanged() {
						t.Fatal("dot modified a single-element input")
					}
				})
			}
		}
	}
}

func TestDotSpanEmptyAndAliasedVectors(t *testing.T) {
	for _, incX := range []int{1, -1, 3, -3} {
		for _, incY := range []int{1, -1, 3, -3} {
			methods, _ := dotSpanTestMethods(nil, nil)
			for _, method := range methods {
				t.Run(fmt.Sprintf("%s/empty/incX=%d/incY=%d", method.name, incX, incY), func(t *testing.T) {
					if got := method.run(0, incX, incY); got != method.bias {
						t.Errorf("empty vectors: got %v, want %v", got, method.bias)
					}
				})
			}
		}
	}
	x := []float64{1, -2, 3}
	methods, unchanged := dotSpanTestMethods(x, x)
	for _, method := range methods {
		t.Run(method.name+"/alias", func(t *testing.T) {
			if got := method.run(3, 1, -1); got != 10+method.bias {
				t.Errorf("aliased vectors: got %v, want %v", got, 10+method.bias)
			}
			if !unchanged() {
				t.Fatal("dot modified an aliased input")
			}
		})
	}
}

func TestDotSpanValidationOrder(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	minInt := -maxInt - 1
	for _, test := range []struct {
		name       string
		n          int
		incX, incY int
		x, y       []float64
		want       string
	}{
		{name: "ZeroXBeforeYAndN", n: -1, incX: 0, incY: 0, want: zeroIncX},
		{name: "ZeroYBeforeN", n: -1, incX: 1, incY: 0, want: zeroIncY},
		{name: "EmptyStillChecksX", n: 0, incX: 0, incY: 1, want: zeroIncX},
		{name: "EmptyStillChecksY", n: 0, incX: 1, incY: 0, want: zeroIncY},
		{name: "NegativeN", n: -1, incX: 1, incY: 1, want: nLT0},
		{name: "ShortXBeforeY", n: 1, incX: 1, incY: 1, want: shortX},
		{name: "ShortY", n: 1, incX: 1, incY: 1, x: []float64{1}, want: shortY},
		{name: "StridedShortX", n: 2, incX: -3, incY: 3, x: []float64{1}, y: []float64{1}, want: shortX},
		{name: "StridedShortY", n: 2, incX: 3, incY: -3, x: []float64{1, 0, 0, 1}, y: []float64{1}, want: shortY},
		{name: "MinIntShortXBeforeY", n: 2, incX: minInt, incY: minInt, x: []float64{1}, y: []float64{1}, want: shortX},
		{name: "MinIntShortY", n: 2, incX: 1, incY: minInt, x: []float64{1, 1}, y: []float64{1}, want: shortY},
		{name: "MaxIntShortXBeforeY", n: 2, incX: maxInt, incY: maxInt, x: []float64{1}, y: []float64{1}, want: shortX},
		{name: "MaxIntShortY", n: 2, incX: 1, incY: maxInt, x: []float64{1, 1}, y: []float64{1}, want: shortY},
		{name: "ProductOverflowXBeforeY", n: 3, incX: minInt, incY: minInt, x: []float64{1}, y: []float64{1}, want: shortX},
		{name: "ProductOverflowY", n: 3, incX: 1, incY: minInt, x: []float64{1, 1, 1}, y: []float64{1}, want: shortY},
	} {
		methods, _ := dotSpanTestMethods(test.x, test.y)
		for _, method := range methods {
			t.Run(method.name+"/"+test.name, func(t *testing.T) {
				defer func() {
					if got := recover(); got != test.want {
						t.Errorf("unexpected panic: got %v, want %q", got, test.want)
					}
				}()
				method.run(test.n, test.incX, test.incY)
			})
		}
	}
}

func dotSpanStrided(values []float64, inc int) []float64 {
	absInc := inc
	if absInc < 0 {
		absInc = -absInc
	}
	x := make([]float64, (len(values)-1)*absInc+1)
	for i := range x {
		x[i] = 1000
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

type dotSpanTestMethod struct {
	name string
	run  func(n, incX, incY int) complex128
	bias complex128
}

func dotSpanTestMethods(x, y []float64) ([]dotSpanTestMethod, func() bool) {
	aliased := len(x) != 0 && len(x) == len(y) && &x[0] == &y[0]
	x, y = slices.Clone(x), slices.Clone(y)
	x32, y32 := make([]float32, len(x)), make([]float32, len(y))
	x64c, y64c := make([]complex64, len(x)), make([]complex64, len(y))
	x128c, y128c := make([]complex128, len(x)), make([]complex128, len(y))
	for i, v := range x {
		x32[i], x64c[i], x128c[i] = float32(v), complex(float32(v), 0), complex(v, 0)
	}
	for i, v := range y {
		y32[i], y64c[i], y128c[i] = float32(v), complex(float32(v), 0), complex(v, 0)
	}
	if aliased {
		y, y32, y64c, y128c = x, x32, x64c, x128c
	}
	x0, y0 := slices.Clone(x), slices.Clone(y)
	x320, y320 := slices.Clone(x32), slices.Clone(y32)
	x64c0, y64c0 := slices.Clone(x64c), slices.Clone(y64c)
	x128c0, y128c0 := slices.Clone(x128c), slices.Clone(y128c)
	impl := Implementation{}
	methods := []dotSpanTestMethod{
		{name: "Ddot", run: func(n, incX, incY int) complex128 { return complex(impl.Ddot(n, x, incX, y, incY), 0) }},
		{name: "Sdot", run: func(n, incX, incY int) complex128 { return complex(float64(impl.Sdot(n, x32, incX, y32, incY)), 0) }},
		{name: "Dsdot", run: func(n, incX, incY int) complex128 { return complex(impl.Dsdot(n, x32, incX, y32, incY), 0) }},
		{name: "Sdsdot", bias: 0.5, run: func(n, incX, incY int) complex128 {
			return complex(float64(impl.Sdsdot(n, 0.5, x32, incX, y32, incY)), 0)
		}},
		{name: "Zdotc", run: func(n, incX, incY int) complex128 { return impl.Zdotc(n, x128c, incX, y128c, incY) }},
		{name: "Zdotu", run: func(n, incX, incY int) complex128 { return impl.Zdotu(n, x128c, incX, y128c, incY) }},
		{name: "Cdotc", run: func(n, incX, incY int) complex128 { return complex128(impl.Cdotc(n, x64c, incX, y64c, incY)) }},
		{name: "Cdotu", run: func(n, incX, incY int) complex128 { return complex128(impl.Cdotu(n, x64c, incX, y64c, incY)) }},
	}
	unchanged := func() bool {
		return slices.Equal(x, x0) && slices.Equal(y, y0) && slices.Equal(x32, x320) && slices.Equal(y32, y320) &&
			slices.Equal(x64c, x64c0) && slices.Equal(y64c, y64c0) && slices.Equal(x128c, x128c0) && slices.Equal(y128c, y128c0)
	}
	return methods, unchanged
}
