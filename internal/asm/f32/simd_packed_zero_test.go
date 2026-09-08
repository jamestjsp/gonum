// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"testing"
)

// The retained private entry supplies the previous 128-bit short algorithm.
// Exact finite bits catch changes that a relative-error comparison would hide
// when large terms cancel and leave a small, representable residual.
func TestSIMDPackedZeroDotGrouping(t *testing.T) {
	for n := 16; n < 32; n++ {
		for _, pattern := range []string{"finite", "cancellation", "negative-zero", "subnormal", "normal-boundary", "exceptional"} {
			x, y := make([]float32, n), make([]float32, n)
			for i := range x {
				x[i], y[i] = float32(i%11-5)/16, float32(i%7-3)/8
			}
			if pattern != "finite" {
				clear(x)
				for i := range y {
					y[i] = 1
				}
			}
			switch pattern {
			case "cancellation":
				x[0], x[4] = 0.75*math.MaxFloat32, -0.75*math.MaxFloat32
				x[8], x[12] = 1, 2
				for i := 16; i < n; i++ {
					x[i] = float32(i%3-1) / 4
				}
			case "negative-zero":
				for i := range x {
					x[i] = math.Float32frombits(1 << 31)
				}
			case "subnormal":
				for i := range x {
					bits := uint32(i%7 + 1)
					if i&1 != 0 {
						bits |= 1 << 31
					}
					x[i] = math.Float32frombits(bits)
					y[i] = []float32{0.5, 1, 2, -1}[i&3]
				}
			case "normal-boundary":
				for i := range x {
					bits := uint32(0x00800000 + i%3 - 1)
					if i&1 != 0 {
						bits |= 1 << 31
					}
					x[i] = math.Float32frombits(bits)
					y[i] = []float32{0.5, 1, -1, 2}[i&3]
				}
			case "exceptional":
				x[0], x[4] = float32(math.Inf(1)), float32(math.Inf(-1))
			}
			check := func(name string, got, want float32) {
				if math.IsNaN(float64(got)) && math.IsNaN(float64(want)) {
					return
				}
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Errorf("%s n=%d pattern=%s: got=%v (%08x), want=%v (%08x)", name, n, pattern, got, math.Float32bits(got), want, math.Float32bits(want))
				}
			}
			check("Dot", DotUnitarySIMD(x, y), dotUnitaryPortableEntrySIMD(x, y))
		}
	}
}

// Every 16-bit signed-zero mask is exercised at every changed length, repeating
// the mask above 16 values. This supplements the finite/cancellation cases; the
// proof that a cannot be negative zero applies independently to every lane.
func TestSIMDPackedZeroDotSigns(t *testing.T) {
	for n := 16; n < 32; n++ {
		x, y := make([]float32, n), make([]float32, n)
		for i := range y {
			y[i] = 1
		}
		for mask := uint32(0); mask < 1<<16; mask++ {
			for i := range x {
				x[i] = math.Float32frombits(((mask >> (i & 15)) & 1) << 31)
			}
			got, want := DotUnitarySIMD(x, y), dotUnitaryPortableEntrySIMD(x, y)
			if math.Float32bits(got) != math.Float32bits(want) {
				t.Fatalf("n%d mask%04x: got%08x want%08x", n, mask, math.Float32bits(got), math.Float32bits(want))
			}
		}
	}
}
