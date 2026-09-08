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

// Finite dyadic fixtures have exact products in both the scalar and paired
// kernels. Verify every gap and both public entries across their admission paths.
func TestSIMDComplexEqualStrideAxpy(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 4096} {
		for _, steps := range [][3]int{{2, 2, 2}, {3, 3, 3}, {7, 7, 7}, {-3, -3, -3}, {2, 3, 5}, {-3, 2, 7}, {0, 0, 0}} {
			t.Run(fmt.Sprintf("n=%d/steps=%v", n, steps), func(t *testing.T) {
				makeStream := func(step, salt int) ([]complex64, uintptr) {
					abs := step
					if abs < 0 {
						abs = -abs
					}
					span := 0
					if n > 0 {
						span = (n - 1) * abs
					}
					v := make([]complex64, span+5)
					for i := range v {
						v[i] = complex(float32((i+salt)%13-6)/16, float32((i+salt)%7-3)/8)
					}
					start := 2
					if step < 0 {
						start += span
					}
					return v, uintptr(start)
				}
				x, ix := makeStream(steps[0], 0)
				y, iy := makeStream(steps[1], 3)
				dst, id := makeStream(steps[2], 5)
				originalX, originalY := slices.Clone(x), slices.Clone(y)
				incX, incY, incD := uintptr(steps[0]), uintptr(steps[1]), uintptr(steps[2])
				want := slices.Clone(dst)
				for j, px, py, pd := 0, ix, iy, id; j < n; j, px, py, pd = j+1, px+incX, py+incY, pd+incD {
					want[pd] = (0.5-0.25i)*x[px] + y[py]
				}
				AxpyIncToSIMD(dst, incD, id, 0.5-0.25i, x, y, uintptr(n), incX, incY, ix, iy)
				if !slices.Equal(dst, want) || !slices.Equal(x, originalX) || !slices.Equal(y, originalY) {
					t.Fatal("AxpyIncTo changed values, inputs, or gaps")
				}
				want = slices.Clone(y)
				for j, px, py := 0, ix, iy; j < n; j, px, py = j+1, px+incX, py+incY {
					want[py] = (0.5-0.25i)*x[px] + want[py]
				}
				AxpyIncSIMD(0.5-0.25i, x, y, uintptr(n), incX, incY, ix, iy)
				if !slices.Equal(y, want) || !slices.Equal(x, originalX) {
					t.Fatal("AxpyInc changed values, inputs, or gaps")
				}
			})
		}
	}
}

// Use the established checked pair loop as the IEEE reference. Scalar Go
// complex64 products have wider intermediates and are not the paired-body oracle.
func TestSIMDComplexEqualStrideAxpyIEEE(t *testing.T) {
	if !complexNativeSIMD() {
		t.Skip("native complex kernel unavailable")
	}
	zero := float32(0)
	values := []complex64{
		complex(zero, float32(math.Copysign(0, -1))),
		complex(float32(math.MaxFloat32), float32(math.SmallestNonzeroFloat32)),
		complex(float32(0.52*math.MaxFloat32), float32(0.13*math.MaxFloat32)),
		complex(float32(math.Inf(1)), float32(math.Inf(-1))),
		complex(float32(math.NaN()), -1),
	}
	for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33} {
		for _, step := range []int{2, -3} {
			for _, alpha := range []complex64{0, 0.5 - 0.25i, 2 + 0.5i, complex(float32(math.Inf(1)), 0)} {
				abs := step
				if abs < 0 {
					abs = -abs
				}
				length := 1 + (n-1)*abs
				start := uintptr(0)
				if step < 0 {
					start = uintptr(length - 1)
				}
				x, y := make([]complex64, length), make([]complex64, length)
				for i := range x {
					x[i], y[i] = values[i%len(values)], values[(i+2)%len(values)]
				}
				for _, alias := range []string{"none", "x", "y", "both", "in-place", "in-place-both"} {
					gx, gy := slices.Clone(x), slices.Clone(y)
					wx, wy := slices.Clone(x), slices.Clone(y)
					got, want := slices.Clone(y), slices.Clone(y)
					switch alias {
					case "x":
						got, want = gx, wx
					case "y", "in-place":
						got, want = gy, wy
					case "both", "in-place-both":
						gy, got, wy, want = gx, gx, wx, wx
					}
					inc := uintptr(step)
					complexAxpyIncCheckedSIMD(want, inc, start, alpha, wx, wy, uintptr(n), inc, inc, start, start)
					if alias == "in-place" || alias == "in-place-both" {
						AxpyIncSIMD(alpha, gx, gy, uintptr(n), inc, inc, start, start)
					} else {
						AxpyIncToSIMD(got, inc, start, alpha, gx, gy, uintptr(n), inc, inc, start, start)
					}
					for i, v := range got {
						for part, pair := range [][2]float32{{real(v), real(want[i])}, {imag(v), imag(want[i])}} {
							if math.Float32bits(pair[0]) != math.Float32bits(pair[1]) && !(math.IsNaN(float64(pair[0])) && math.IsNaN(float64(pair[1]))) {
								t.Fatalf("n=%d step=%d alpha=%v alias=%s index=%d part=%d got=%v want=%v", n, step, alpha, alias, i, part, v, want[i])
							}
						}
					}
				}
			}
		}
	}
}
