// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"fmt"
	"math"
	"simd"
	"testing"
)

func TestC64UnitaryCachedPredicate(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	lengths := []int{0, 1, 3, 4, 7, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 510, 511, 512, 513, 4096, maxInt - 1, maxInt}
	for n := 0; n <= 1024; n++ {
		lengths = append(lengths, n)
	}
	width, native := simd.VectorBitSize(), complexNativeSIMD()
	expected := -1
	if native && width >= 256 {
		expected = 511
		if width == 256 {
			expected = maxInt
		}
	}
	if nativeDotUnitaryMaxSIMD != expected {
		t.Fatalf("width=%d native=%t bound=%d want=%d", width, native, nativeDotUnitaryMaxSIMD, expected)
	}
	for _, n := range lengths {
		old := n >= 4 && native && width >= 256 && (width == 256 || n < 512)
		cached := n >= 4 && n <= nativeDotUnitaryMaxSIMD
		if old != cached {
			t.Fatalf("n=%d width=%d native=%t old=%t cached=%t", n, width, native, old, cached)
		}
	}
	// This finite partition includes all changes of truth value (4,512 and
	// MaxInt), without constructing invalid or impractically large slices.
	// Also check hypothetical widths: the old expression did not require512.
	for _, isNative := range []bool{false, true} {
		for _, w := range []int{0, 128, 255, 256, 257, 511, 512, 1024} {
			bound := -1
			if isNative && w >= 256 {
				bound = 511
				if w == 256 {
					bound = maxInt
				}
			}
			for _, n := range lengths {
				old := n >= 4 && isNative && w >= 256 && (w == 256 || n < 512)
				if got := n >= 4 && n <= bound; got != old {
					t.Fatalf("partition n=%d w=%d native=%t: %t != %t", n, w, isNative, got, old)
				}
			}
			if isNative && w == 256 && !(maxInt >= 4 && maxInt <= bound) {
				t.Fatal("MaxInt lost by exclusive upper bound")
			}
		}
	}
}

func c64CacheAssertBits(t *testing.T, got, want complex64) {
	t.Helper()
	for _, p := range [][2]float32{{real(got), real(want)}, {imag(got), imag(want)}} {
		if math.IsNaN(float64(p[1])) && math.IsNaN(float64(p[0])) {
			continue
		}
		if math.Float32bits(p[0]) != math.Float32bits(p[1]) {
			t.Fatalf("got %v [%08x,%08x], want %v [%08x,%08x]", got, math.Float32bits(real(got)), math.Float32bits(imag(got)), want, math.Float32bits(real(want)), math.Float32bits(imag(want)))
		}
	}
}

func TestC64UnitaryCachedResultParity(t *testing.T) {
	m := float32(0.75 * math.MaxFloat32)
	negzero := math.Float32frombits(1 << 31)
	cases := [][]complex64{
		{complex(0, negzero), complex(negzero, 0), complex(negzero, negzero)},
		{complex(math.SmallestNonzeroFloat32, 0), complex(-math.SmallestNonzeroFloat32, negzero)},
		{complex(math.Float32frombits(0x7fc00001), 1), complex(2, 3)},
		{complex(float32(math.Inf(1)), 0), complex(float32(math.Inf(-1)), 1)},
		{complex(m, m), complex(-m, m), complex(m, -m), complex(-m, -m)},
		{1 + 2i, -3 + 4i, 5 - 6i, -7 - 8i},
	}
	for _, n := range []int{0, 1, 2, 3, 4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 510, 511, 512, 513, 4096} {
		for ci, values := range cases {
			for _, alias := range []bool{false, true} {
				t.Run(fmt.Sprintf("n=%d/case=%d/alias=%t", n, ci, alias), func(t *testing.T) {
					x, y := make([]complex64, n), make([]complex64, n)
					for i := range x {
						x[i] = values[i%len(values)]
						y[i] = complex(float32(i%3-1), float32(i%5-2))
					}
					if alias {
						y = x
					}
					for _, conjugate := range []bool{false, true} {
						original, current := dotuUnitaryUncachedReference, DotuUnitarySIMD
						if conjugate {
							original, current = dotcUnitaryUncachedReference, DotcUnitarySIMD
						}
						c64CacheAssertBits(t, current(x, y), original(x, y))
					}
				})
			}
		}
	}
}

func TestC64UnitaryCachedYBounds(t *testing.T) {
	didPanic := func(fn func([]complex64, []complex64) complex64, x, y []complex64) (panicked bool) {
		defer func() { panicked = recover() != nil }()
		fn(x, y)
		return false
	}
	for _, n := range []int{1, 2, 3, 4, 7, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 510, 511, 512, 513} {
		for _, spareCapacity := range []bool{false, true} {
			x := make([]complex64, n)
			capY := n - 1
			if spareCapacity {
				capY = n
			}
			y := make([]complex64, n-1, capY)
			for _, pair := range [][2]func([]complex64, []complex64) complex64{{dotcUnitaryUncachedReference, DotcUnitarySIMD}, {dotuUnitaryUncachedReference, DotuUnitarySIMD}} {
				want, got := didPanic(pair[0], x, y), didPanic(pair[1], x, y)
				if want != got {
					t.Fatalf("n=%d spare=%t panic got=%t old=%t", n, spareCapacity, got, want)
				}
				if !spareCapacity && !got {
					t.Fatalf("n=%d: neither entry rejected insufficient backing", n)
				}
			}
		}
	}
}
