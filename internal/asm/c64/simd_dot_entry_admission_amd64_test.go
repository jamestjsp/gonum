// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// Dyadic products and sums below are exactly representable in binary32. The
// independent complex128 oracle does not mirror SIMD grouping or admission.
func TestSIMDComplexDotEntryAdmission(t *testing.T) {
	funcs := []struct {
		name      string
		conjugate bool
		dot       func([]complex64, []complex64, uintptr, uintptr, uintptr, uintptr, uintptr) complex64
	}{{"Dotc", true, DotcIncSIMD}, {"Dotu", false, DotuIncSIMD}}
	for _, fn := range funcs {
		if got := fn.dot(nil, nil, 0, ^uintptr(0), 0, ^uintptr(0), ^uintptr(0)); math.Float32bits(real(got)) != 0 || math.Float32bits(imag(got)) != 0 {
			t.Fatalf("%s empty invalid starts: %v", fn.name, got)
		}
		for _, n := range []int{1, 2, 3, 4, 7, 8, 31, 32, 33, 63, 64, 65} {
			for _, inc := range [][2]int{{0, 0}, {0, 3}, {3, 0}, {1, 1}, {1, 3}, {2, 7}, {3, 2}, {-1, 2}, {2, -3}, {-2, -7}} {
				span := func(step int) (size, start int) {
					if step < 0 {
						return (n-1)*(-step) + 5, (n-1)*(-step) + 2
					}
					return (n-1)*step + 5, 2
				}
				nx, ix := span(inc[0])
				ny, iy := span(inc[1])
				for _, alias := range []string{"separate", "same", "shifted"} {
					t.Run(fmt.Sprintf("%s/n=%d/inc=%d,%d/alias=%s", fn.name, n, inc[0], inc[1], alias), func(t *testing.T) {
						data := make([]complex64, max(nx, ny+1))
						x, y := data[:nx], make([]complex64, ny)
						if alias == "same" {
							y = data[:ny]
						}
						if alias == "shifted" {
							y = data[1 : ny+1]
						}
						for i := range data {
							data[i] = complex(float32(i%11-5)/8, float32(i%7-3)/16)
						}
						if alias == "separate" {
							for i := range y {
								y[i] = complex(float32(i%13-6)/16, float32(i%5-2)/8)
							}
						}
						originalX, originalY := slices.Clone(x), slices.Clone(y)
						var want complex128
						for k := 0; k < n; k++ {
							xv := complex128(x[ix+k*inc[0]])
							if fn.conjugate {
								xv = complex(real(xv), -imag(xv))
							}
							want += xv * complex128(y[iy+k*inc[1]])
						}
						got := fn.dot(x, y, uintptr(n), uintptr(inc[0]), uintptr(inc[1]), uintptr(ix), uintptr(iy))
						if complex128(got) != want {
							t.Fatalf("got=%v want=%v", got, want)
						}
						if !slices.Equal(x, originalX) || !slices.Equal(y, originalY) {
							t.Fatal("dot modified readonly inputs")
						}
					})
				}
			}
		}
		cases := []struct{ n, sx, sy, ix, iy uintptr }{
			{1, 2, 2, 4, 0}, {1, 2, 2, 0, 4},
			{3, 3, 2, 0, 0}, {3, ^uintptr(0), 2, 0, 0},
			{4, ^uintptr(0) >> 1, 2, 1, 0},
			{2, 0, 0, ^uintptr(0), 0}, {2, 1, 1, ^uintptr(0), 0},
		}
		for i, tc := range cases {
			t.Run(fmt.Sprintf("%s/invalid=%d", fn.name, i), func(t *testing.T) {
				x, y := []complex64{1, 2, 3, 4}, []complex64{4, 3, 2, 1}
				beforeX, beforeY := slices.Clone(x), slices.Clone(y)
				defer func() {
					if recover() == nil {
						t.Error("invalid stream did not panic")
					}
					if !slices.Equal(x, beforeX) || !slices.Equal(y, beforeY) {
						t.Error("invalid readonly stream modified input")
					}
				}()
				fn.dot(x, y, tc.n, tc.sx, tc.sy, tc.ix, tc.iy)
			})
		}
	}
}
