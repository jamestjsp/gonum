// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math"
	"reflect"
	"testing"
)

func TestSIMDComplexStrideBounds(t *testing.T) {
	for _, test := range []struct {
		length        int
		n, inc, index uintptr
		want          bool
	}{
		{0, 0, ^uintptr(0), ^uintptr(0), true},
		{0, 1, 0, 0, false},
		{16, 1, ^uintptr(0), 15, true},
		{16, 1, 1, 16, false},
		{16, 8, 2, 1, true},
		{16, 9, 2, 0, false},
		{16, 8, ^uintptr(1), 15, true},
		{16, 9, ^uintptr(1), 15, false},
		{16, ^uintptr(0), 0, 15, true},
		{16, ^uintptr(0), 2, 0, false},
		{16, ^uintptr(0), ^uintptr(1), 15, false},
		{16, 3, 1 << 63, 15, false},
		{16, 3, 1<<63 - 1, 0, false},
	} {
		got := complexStrideInBoundsSIMD(test.length, test.n, test.inc, test.index)
		if got != test.want {
			t.Errorf("len=%d n=%d inc=%d index=%d: got %t want %t", test.length, test.n, test.inc, test.index, got, test.want)
		}
	}
}

// An invalid native span must retain the checked native loop's ordered writes
// before panic. The portable fallback stages whole blocks before writing.
func TestSIMDComplexInvalidStrideWrites(t *testing.T) {
	if !complexNativeSIMD() {
		t.Skip("native complex kernel unavailable")
	}
	for _, test := range []struct {
		n, incX, incY, incDst, ix, iy, idst uintptr
	}{
		{8, 2, 1, 1, 0, 0, 0},
		{8, 1, 2, 1, 0, 0, 0},
		{8, 1, 1, 2, 0, 0, 0},
		{4, ^uintptr(1), 1, 1, 5, 0, 0},
	} {
		x, y, want, got := make([]complex128, 11), make([]complex128, 11), make([]complex128, 11), make([]complex128, 11)
		for i := range x {
			x[i], y[i] = complex(float64(i+1), -0.5), 2+0.25i
			want[i], got[i] = -123, -123
		}
		panics := func(f func()) (didPanic bool) {
			defer func() { didPanic = recover() != nil }()
			f()
			return false
		}
		if !panics(func() {
			ix, iy, idst := test.ix, test.iy, test.idst
			for i := uintptr(0); i < test.n; i++ {
				want[idst] = (0.25-0.5i)*x[ix] + y[iy]
				ix, iy, idst = ix+test.incX, iy+test.incY, idst+test.incDst
			}
		}) {
			t.Fatal("invalid scalar fixture did not panic")
		}
		if !panics(func() {
			AxpyIncToSIMD(got, test.incDst, test.idst, 0.25-0.5i, x, y, test.n, test.incX, test.incY, test.ix, test.iy)
		}) {
			t.Errorf("invalid SIMD span did not panic: %+v", test)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("invalid span changed writes before panic: %+v\ngot %v\nwant %v", test, got, want)
		}
	}
}

// Every complex product cancels its real components. Alternating groups keep
// the established four accumulators finite, but component sums overflow. A
// sequential or portable-width retry alone can also overflow this sequence.
func TestSIMDComplexStridedComponentRecovery(t *testing.T) {
	if !complexNativeSIMD() {
		t.Skip("native complex kernel unavailable")
	}
	const n = 16
	m := 0.4 * math.MaxFloat64
	for _, conjugate := range []bool{false, true} {
		x, y := make([]complex128, 2*n), make([]complex128, 3*n)
		for i := 0; i < n; i++ {
			sign := 1.0
			if i/4%2 != 0 {
				sign = -1
			}
			x[2*i] = complex(m, sign*m)
			if conjugate {
				sign = -sign
			}
			y[3*i] = complex(1, sign)
		}
		got := DotuIncSIMD(x, y, n, 2, 3, 0, 0)
		if conjugate {
			got = DotcIncSIMD(x, y, n, 2, 3, 0, 0)
		}
		if got != 0 {
			t.Errorf("conjugate=%t: got %v want 0", conjugate, got)
		}
	}
}
