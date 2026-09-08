// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"reflect"
	"simd"
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
		x, y, want, got := make([]complex64, 11), make([]complex64, 11), make([]complex64, 11), make([]complex64, 11)
		for i := range x {
			x[i], y[i] = complex(float32(i+1), -0.5), 2+0.25i
			want[i], got[i] = -123, -123
		}
		panics := func(f func()) (didPanic bool) {
			defer func() { didPanic = recover() != nil }()
			f()
			return false
		}
		if !panics(func() {
			complexAxpyIncCheckedSIMD(want, test.incDst, test.idst, 0.25-0.5i, x, y, test.n, test.incX, test.incY, test.ix, test.iy)
		}) {
			t.Fatal("invalid checked fixture did not panic")
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
	for _, n := range []int{16, 64, 128} {
		m := float32(0.4 * math.MaxFloat32)
		for _, conjugate := range []bool{false, true} {
			x, y := make([]complex64, 2*n), make([]complex64, 3*n)
			for i := 0; i < n; i++ {
				sign := float32(1)
				if i/4%2 != 0 {
					sign = -1
				}
				x[2*i] = complex(m, sign*m)
				if conjugate {
					sign = -sign
				}
				y[3*i] = complex(1, sign)
			}
			got := DotuIncSIMD(x, y, uintptr(n), 2, 3, 0, 0)
			if conjugate {
				got = DotcIncSIMD(x, y, uintptr(n), 2, 3, 0, 0)
			}
			if got != 0 {
				t.Errorf("conjugate=%t: got %v want 0", conjugate, got)
			}
		}
	}
}

// The eight-complex short loop sums equal-sign terms before cancellation. The
// previous four-complex grouping cancels them first, so retain that cold retry.
func TestSIMDComplexShortWideGroupingRecovery(t *testing.T) {
	if !complexNativeSIMD() || simd.VectorBitSize() < 256 {
		t.Skip("wide short kernel unavailable")
	}
	m := float32(0.75 * math.MaxFloat32)
	x, y := make([]complex64, 16), make([]complex64, 16)
	for i := range y {
		y[i] = 1
	}
	x[0], x[4], x[8], x[12] = complex(m, 0), complex(-m, 0), complex(m, 0), complex(-m, 0)
	for _, conjugate := range []bool{false, true} {
		raw := complexDotShortWideSIMD(x, y, conjugate)
		if math.Float32bits(real(raw))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(raw))&0x7f800000 != 0x7f800000 {
			t.Fatalf("conjugate=%t: wide fixture did not overflow: %v", conjugate, raw)
		}
		if got := portableDotUnitarySIMD(x, y, conjugate); got != 0 {
			t.Errorf("conjugate=%t: got %v want 0", conjugate, got)
		}
	}
}

// Go scalar complex64 multiplication widens its products before rounding the
// complex result to float32. Native tails must preserve the established scalar
// remainder positions, where separate float32 products can overflow instead.
func TestSIMDComplexAxpyScalarTailClassification(t *testing.T) {
	if !complexNativeSIMD() {
		t.Skip("native complex kernel unavailable")
	}
	alpha := complex(float32(2), float32(0.5))
	value := complex(float32(0.52*math.MaxFloat32), float32(0.13*math.MaxFloat32))
	for n := 4; n <= 65; n++ {
		x, y := make([]complex64, n), make([]complex64, n)
		x[n-1] = value
		want, got := make([]complex64, n), make([]complex64, n)
		portableAxpyUnitaryToSIMD(want, alpha, x, y)
		AxpyUnitaryToSIMD(got, alpha, x, y)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("unitary n=%d: last=%v want %v", n, got[n-1], want[n-1])
		}
		for _, step := range []int{-3, 3} {
			length := 3*(n-1) + 1
			x, y = make([]complex64, length), make([]complex64, length)
			want, got = make([]complex64, length), make([]complex64, length)
			index := 0
			if step < 0 {
				index = length - 1
			}
			last := index + (n-1)*step
			x[last] = value
			complexAxpyIncCheckedSIMD(want, uintptr(step), uintptr(index), alpha, x, y, uintptr(n), uintptr(step), uintptr(step), uintptr(index), uintptr(index))
			AxpyIncToSIMD(got, uintptr(step), uintptr(index), alpha, x, y, uintptr(n), uintptr(step), uintptr(step), uintptr(index), uintptr(index))
			if !reflect.DeepEqual(got, want) {
				t.Errorf("stride n=%d inc=%d: last=%v want %v", n, step, got[last], want[last])
			}
		}
	}
}
