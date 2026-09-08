// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math32

import (
	"math"
	"testing"
)

// TestSqrtIEEE compares exact results against the independent integer algorithm
// in math_test.go, including cases ordinary random real values seldom exercise.
func TestSqrtIEEE(t *testing.T) {
	inputs := []uint32{0, 0x80000000, 1, 2, 0x007fffff, 0x00800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001, 0x7f800001}
	for exponent := uint32(0); exponent < 255; exponent++ {
		for _, fraction := range []uint32{0, 1, 0x155555, 0x400000, 0x7fffff} {
			inputs = append(inputs, exponent<<23|fraction)
		}
	}
	// A fixed bit-pattern sequence covers both signs without depending on a
	// random seed or concentrating the input distribution near zero.
	bits := uint32(1)
	for range 4096 {
		bits = bits*1664525 + 1013904223
		inputs = append(inputs, bits)
	}
	for _, input := range inputs {
		x := math.Float32frombits(input)
		got, want := Sqrt(x), sqrt(x)
		if !alike(got, want) {
			t.Errorf("sqrt(%08x): got %08x want %08x", input, math.Float32bits(got), math.Float32bits(want))
		}
	}
}
