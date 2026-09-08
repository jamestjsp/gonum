// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"simd"
	"simd/archsimd"
	"testing"
	"unsafe"
)

func TestSIMDStridedPairBits(t *testing.T) {
	if simd.Emulated() || !archsimd.X86.AVX() {
		t.Skip("native AVX pair packing")
	}
	for _, inc := range []int{1, 2, 3, 7, 63} {
		x := make([]uint32, 1+3*inc)
		want := [4]uint32{0x80000000, 0x7fc12345, 0xff800000, 0x00000001}
		for i, bits := range want {
			x[i*inc] = bits
		}
		got := gatherDotStridedPair4(unsafe.Pointer(&x[0]), uintptr(inc*4)).AsUint32x4()
		var lanes [4]uint32
		got.StoreArray(&lanes)
		for i, bits := range want {
			if lanes[i] != bits {
				t.Fatalf("inc=%d lane=%d: got%08x want%08x", inc, i, lanes[i], bits)
			}
		}
	}
}

func TestSIMDDotIncPairGrouping(t *testing.T) {
	if simd.Emulated() || !archsimd.X86.AVX() {
		t.Skip("native AVX pair packing")
	}
	for _, n := range []int{4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 4096} {
		for _, inc := range [][2]int{{1, 3}, {2, 7}, {3, 3}, {7, 2}, {63, 16}} {
			for mode := 0; mode < 3; mode++ {
				x := make([]float32, 2+(n-1)*inc[0])
				y := make([]float32, 3+(n-1)*inc[1])
				for i := 0; i < n; i++ {
					x[1+i*inc[0]] = float32((i*7)%31-15) / 16
					y[2+i*inc[1]] = float32((i*11)%23-11) / 8
					if mode == 1 {
						x[1+i*inc[0]] = math.Float32frombits(0x80000000)
						y[2+i*inc[1]] = 1
					}
					if mode == 2 {
						x[1+i*inc[0]] = 0
						y[2+i*inc[1]] = 1
						if i%16 == 0 {
							x[1+i*inc[0]] = 0.75 * math.MaxFloat32
						} else if i%16 == 8 {
							x[1+i*inc[0]] = -0.75 * math.MaxFloat32
						}
					}
				}
				nn, sx, sy := uintptr(n), uintptr(inc[0]), uintptr(inc[1])
				want := dotIncPositiveSIMD(x, y, nn, sx, sy, 1, 2)
				if math.Float32bits(want)&0x7f800000 == 0x7f800000 {
					want = dotIncPortableSIMD(x, y, nn, sx, sy, 1, 2)
					if math.Float32bits(want)&0x7f800000 == 0x7f800000 {
						want = dotIncSequentialSIMD(x, y, nn, sx, sy, 1, 2)
					}
				}
				got := DotIncSIMD(x, y, nn, sx, sy, 1, 2)
				if math.Float32bits(got) != math.Float32bits(want) && !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) {
					t.Fatalf("n=%d inc=%v mode=%d: got%g want%g", n, inc, mode, got, want)
				}
			}
		}
	}
}
