// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"math"
	"simd"
	"simd/archsimd"
	"testing"
)

// The old packed oracle deliberately covers only16..31. This independent
// scalar lane model fills the newly pointer-rewritten4..15 gap.
func sumPointerLowerLaneReference(x []float32) float32 {
	if len(x) < 4 || len(x) >= 16 {
		panic("lower short oracle requires4..15")
	}
	var a, b, c, d [4]float32
	i := 0
	if len(x) >= 8 {
		for j := range a {
			a[j] = x[j] + a[j]
			b[j] = x[4+j] + b[j]
		}
		i = 8
	}
	if len(x)-i >= 4 {
		for j := range c {
			c[j] = x[i+j] + c[j]
		}
		i += 4
	}
	if len(x)-i == 3 {
		d[0] += float32(0)
		for j := 1; j < 4; j++ {
			d[j] += x[i+j-1]
		}
	}
	var acc [4]float32
	for j := range acc {
		acc[j] = (a[j] + b[j]) + (c[j] + d[j])
	}
	sum := (acc[0] + acc[2]) + (acc[1] + acc[3])
	if len(x)-i != 3 {
		for _, v := range x[i:] {
			sum += v
		}
	}
	return sum
}

func sumPointerEqual(t *testing.T, label string, got, want float32) {
	t.Helper()
	if math.IsNaN(float64(got)) && math.IsNaN(float64(want)) {
		return
	}
	if math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("%s got=%08x want=%08x", label, math.Float32bits(got), math.Float32bits(want))
	}
}

func TestSIMDSumPointerLowerIEEE(t *testing.T) {
	tiny := float32(math.SmallestNonzeroFloat32)
	families := [][]float32{
		{0, math.Float32frombits(1 << 31)},
		{1, -1, 0x1p24, 0x1p-24, -0x1p24, 3, tiny, -tiny},
		{.75 * math.MaxFloat32, -.75 * math.MaxFloat32, 0, 0},
		{tiny, -tiny, 2 * tiny, -3 * tiny, 5 * tiny, -8 * tiny},
		{math.Float32frombits(0x007fffff), -math.Float32frombits(0x00800000), math.Float32frombits(0x00800001), -tiny},
	}
	for n := 4; n < 16; n++ {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			check := func(x []float32) {
				t.Helper()
				want := sumPortableEntrySIMD(x)
				got := SumSIMD(x)
				sumPointerEqual(t, "public portable reference", got, want)
				if !simd.Emulated() && archsimd.X86.AVX() {
					ref := sumPointerLowerLaneReference(x)
					sumPointerEqual(t, "unchanged raw helper", sumShortHardwareSIMD(x), ref)
					if math.Float32bits(ref)&0x7f800000 == 0x7f800000 {
						ref = sumOriginalSIMD(x)
						if math.Float32bits(ref)&0x7f800000 == 0x7f800000 {
							ref = sumSequentialSIMD(x)
						}
					}
					sumPointerEqual(t, "same complete recovery", got, ref)
				}
			}
			for _, values := range families {
				for phase := range values {
					for offset := 0; offset < 4; offset++ {
						backing := make([]float32, offset+n+1)
						x := backing[offset : offset+n : offset+n]
						for i := range x {
							x[i] = values[(i+phase)%len(values)]
						}
						check(x)
						for _, special := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345), math.Float32frombits(0x7f812345)} {
							for pos := range x {
								old := x[pos]
								x[pos] = special
								check(x)
								x[pos] = old
							}
						}
					}
				}
			}
		})
	}
}

func TestSIMDSumPointerCapacity(t *testing.T) {
	for n := 4; n < 32; n++ {
		for offset := 0; offset < 4; offset++ {
			for _, extra := range []int{0, 1, 7, 32} {
				t.Run(fmt.Sprintf("n=%d/offset=%d/extra=%d", n, offset, extra), func(t *testing.T) {
					backing := make([]float32, offset+n+extra+1)
					for i := range backing {
						backing[i] = math.Float32frombits(0x7fc10000 | uint32(i))
					}
					x := backing[offset : offset+n : offset+n+extra]
					for i := range x {
						x[i] = float32((i*7+n)%17-8) / 8
					}
					before := append([]float32(nil), backing...)
					want := sumPortableEntrySIMD(x[:n:n])
					sumPointerEqual(t, "capacity-independent public result", SumSIMD(x), want)
					for i := range backing {
						if math.Float32bits(backing[i]) != math.Float32bits(before[i]) {
							t.Fatalf("input/poison changed at%d", i)
						}
					}
				})
			}
		}
	}
}
