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

// Scalar lane bookkeeping records the established 128-bit short tree without
// using architecture vectors or the proposed packed implementation.
func sumPackedZeroLaneReference(x []float32) float32 {
	if len(x) < 16 || len(x) >= 32 {
		panic("reference is defined only for changed short lengths")
	}
	var a, b, c, d [4]float32
	for j := range a {
		a[j] = x[j] + float32(0)
		b[j], c[j], d[j] = x[4+j], x[8+j], x[12+j]
	}
	i := 16
	if len(x) >= 24 {
		for j := range a {
			a[j] = x[16+j] + a[j]
			b[j] = x[20+j] + b[j]
		}
		i = 24
	}
	if len(x)-i >= 4 {
		for j := range c {
			c[j] = x[i+j] + c[j]
		}
		i += 4
	}
	if len(x)-i == 3 {
		// The original complete overlapping load clears its duplicate lane
		// to positive zero and still adds that zero to D[0].
		d[0] += float32(0)
		for j := 1; j < 4; j++ {
			d[j] += x[i+j-1]
		}
	}
	var acc [4]float32
	for j := range acc {
		acc[j] = (a[j] + b[j]) + (c[j] + d[j])
	}
	result := (acc[0] + acc[2]) + (acc[1] + acc[3])
	if len(x)-i != 3 {
		for _, v := range x[i:] {
			result += v
		}
	}
	return result
}

func checkSumPackedZero(t *testing.T, name string, x []float32) {
	t.Helper()
	original := append([]float32(nil), x...)
	want := sumPortableEntrySIMD(x)
	got := SumSIMD(x)
	check := func(label string, got, want float32) {
		t.Helper()
		if math.IsNaN(float64(got)) && math.IsNaN(float64(want)) {
			return
		}
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("%s %s n=%d: got=%g (%08x), want=%g (%08x), x=%v", name, label, len(x), got, math.Float32bits(got), want, math.Float32bits(want), x)
		}
	}
	check("public", got, want)
	if !simd.Emulated() && archsimd.X86.AVX() {
		// Independent grouping evidence before the established public
		// exceptional-result retries, then exact complete recovery order.
		ref := sumPackedZeroLaneReference(x)
		check("raw lane tree", sumShortHardwareSIMD(x), ref)
		if math.Float32bits(ref)&0x7f800000 == 0x7f800000 {
			ref = sumOriginalSIMD(x)
			if math.Float32bits(ref)&0x7f800000 == 0x7f800000 {
				ref = sumSequentialSIMD(x)
			}
		}
		check("recovered lane tree", got, ref)
	}
	for i := range x {
		if math.Float32bits(x[i]) != math.Float32bits(original[i]) {
			t.Fatalf("%s n=%d modified input index%d", name, len(x), i)
		}
	}
}

func TestSIMDSumPackedZeroGrouping(t *testing.T) {
	tiny := float32(math.SmallestNonzeroFloat32)
	negativeZero := math.Float32frombits(1 << 31)
	families := [][]float32{
		{0, negativeZero},
		{1, -1, 0x1p24, 0x1p-24, -0x1p24, 3, tiny, -tiny},
		{.75 * math.MaxFloat32, -.75 * math.MaxFloat32, 0, 0},
		{tiny, -tiny, 2 * tiny, -3 * tiny, 5 * tiny, -8 * tiny},
		{math.Float32frombits(0x007fffff), -math.Float32frombits(0x00800000), math.Float32frombits(0x00800001), -tiny},
	}
	for n := 16; n < 32; n++ {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			for family, values := range families {
				for phase := range values {
					for offset := 0; offset < 4; offset++ {
						backing := make([]float32, n+offset+2)
						for i := range backing {
							backing[i] = -19.25
						}
						x := backing[offset : offset+n : offset+n]
						for i := range x {
							x[i] = values[(i+phase)%len(values)]
						}
						checkSumPackedZero(t, fmt.Sprintf("family%d/phase%d/offset%d", family, phase, offset), x)
						for _, i := range []int{n + offset, n + offset + 1} {
							if backing[i] != -19.25 {
								t.Fatalf("suffix sentinel%d changed", i)
							}
						}
						for _, v := range backing[:offset] {
							if v != -19.25 {
								t.Fatal("prefix sentinel changed")
							}
						}
					}
				}
			}
			// A finite extreme cancellation leaves a small residual in the
			// original lanes; a sequential-first retry would lose it.
			x := make([]float32, n)
			x[0], x[1] = .75*math.MaxFloat32, .75*math.MaxFloat32
			x[4], x[5], x[8], x[12] = -.75*math.MaxFloat32, -.75*math.MaxFloat32, 1, 2
			checkSumPackedZero(t, "finite cancellation", x)
			for _, special := range []float32{float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc12345), math.Float32frombits(0x7f812345)} {
				for pos := range n {
					for i := range x {
						x[i] = float32(i%7-3) / 8
					}
					x[pos] = special
					checkSumPackedZero(t, fmt.Sprintf("special%08x/pos%d", math.Float32bits(special), pos), x)
				}
			}
		})
	}
}

func TestSIMDSumPackedZeroSigns(t *testing.T) {
	for n := 16; n < 32; n++ {
		x := make([]float32, n)
		for mask := uint32(0); mask < 1<<16; mask++ {
			for i := range x {
				x[i] = math.Float32frombits(((mask >> (i & 15)) & 1) << 31)
			}
			got, want := SumSIMD(x), sumPortableEntrySIMD(x)
			if math.Float32bits(got) != math.Float32bits(want) || math.Float32bits(got) != 0 {
				t.Fatalf("n%d mask%04x: got%08x want%08x", n, mask, math.Float32bits(got), math.Float32bits(want))
			}
		}
	}
}
