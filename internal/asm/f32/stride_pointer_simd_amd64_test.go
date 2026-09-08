// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"math"
	"testing"
)

func TestSIMDPositiveStridedSpans(t *testing.T) {
	for _, n := range []int{127, 128, 129, 255, 257, 1025, 4096} {
		for _, incs := range [][3]int{{1, 2, 3}, {2, 2, 2}, {2, 3, 5}, {3, 7, 2}, {7, 16, 3}, {63, 2, 7}} {
			t.Run(fmt.Sprintf("n=%d/incs=%v", n, incs), func(t *testing.T) {
				ix, iy, idst := 3, 5, 7
				x := make([]float32, ix+(n-1)*incs[0]+1)
				y := make([]float32, iy+(n-1)*incs[1]+1)
				dst := make([]float32, idst+(n-1)*incs[2]+1)
				for i := range x {
					x[i] = float32(math.NaN())
				}
				for i := range y {
					y[i] = float32(math.NaN())
				}
				for i := range dst {
					dst[i] = -99
				}
				var dot float32
				for i := 0; i < n; i++ {
					a, b := float32(i%9-4), float32(i%7-3)
					x[ix+i*incs[0]] = a
					y[iy+i*incs[1]] = b
					dot += a * b
				}
				if got := DotIncSIMD(x, y, uintptr(n), uintptr(incs[0]), uintptr(incs[1]), uintptr(ix), uintptr(iy)); got != dot {
					t.Errorf("dot=%v want%v", got, dot)
				}
				if got := DdotIncSIMD(x, y, uintptr(n), uintptr(incs[0]), uintptr(incs[1]), uintptr(ix), uintptr(iy)); got != float64(dot) {
					t.Errorf("ddot=%v want%v", got, dot)
				}
				AxpyIncToSIMD(dst, uintptr(incs[2]), uintptr(idst), 2, x, y, uintptr(n), uintptr(incs[0]), uintptr(incs[1]), uintptr(ix), uintptr(iy))
				for i, v := range dst {
					want := float32(-99)
					if i >= idst && (i-idst)%incs[2] == 0 {
						j := (i - idst) / incs[2]
						want = 2*x[ix+j*incs[0]] + y[iy+j*incs[1]]
					}
					if v != want {
						t.Fatalf("dst[%d]=%v want%v", i, v, want)
					}
				}
				AxpyIncSIMD(2, x, y, uintptr(n), uintptr(incs[0]), uintptr(incs[1]), uintptr(ix), uintptr(iy))
				for i, v := range y {
					if i >= iy && (i-iy)%incs[1] == 0 {
						j := (i - iy) / incs[1]
						want := 2*float32(j%9-4) + float32(j%7-3)
						if v != want {
							t.Fatalf("y[%d]=%v want%v", i, v, want)
						}
					} else if !math.IsNaN(float64(v)) {
						t.Fatalf("modified y gap[%d]=%v", i, v)
					}
				}
			})
		}
	}
}

func TestSIMDPositiveStrideOverflow(t *testing.T) {
	// Both mathematical spans exceed the allocation even though a wrapped
	// uintptr product could make an endpoint look small.
	for _, inc := range []uintptr{1 << 62, 1<<63 - 1} {
		if positiveStrideSIMD(16, 0, 129, inc) {
			t.Errorf("accepted overflowing increment %d", inc)
		}
	}
	if positiveStrideSIMD(16, 0, 0, 1) || positiveStrideSIMD(16, 16, 1, 1) {
		t.Error("accepted empty or out-of-bounds span")
	}
}
