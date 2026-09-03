// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f32

import (
	"slices"
	"testing"
)

func TestSasumUnitarySIMDLengths(t *testing.T) {
	for n := 0; n <= 33; n++ {
		x := make([]float32, n)
		var want float32
		for i := range x {
			x[i] = float32(i%7 - 3)
			if x[i] < 0 {
				want -= x[i]
			} else {
				want += x[i]
			}
		}
		if got := SasumUnitary(x); got != want {
			t.Errorf("n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestLevel1UnitarySIMDLengths(t *testing.T) {
	for n := 0; n <= 33; n++ {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(i - 4)
			y[i] = float32(2*i + 1)
		}

		wantX := slices.Clone(y)
		wantY := slices.Clone(x)
		gotX := slices.Clone(x)
		gotY := slices.Clone(y)
		SwapUnitary(gotX, gotY)
		checkLevel1Pair(t, "swap", n, gotX, gotY, wantX, wantY)

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		rotUnitaryScalar(wantX, wantY, 2, 3)
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotUnitary(gotX, gotY, 2, 3)
		checkLevel1Pair(t, "rot", n, gotX, gotY, wantX, wantY)

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		rotmUnitaryRescalingScalar(wantX, wantY, 2, 3, 4, 5)
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotmUnitaryRescaling(gotX, gotY, 2, 3, 4, 5)
		checkLevel1Pair(t, "rotm rescaling", n, gotX, gotY, wantX, wantY)

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		rotmUnitaryOffDiagonalScalar(wantX, wantY, 3, 4)
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotmUnitaryOffDiagonal(gotX, gotY, 3, 4)
		checkLevel1Pair(t, "rotm off-diagonal", n, gotX, gotY, wantX, wantY)

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		rotmUnitaryDiagonalScalar(wantX, wantY, 2, 5)
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotmUnitaryDiagonal(gotX, gotY, 2, 5)
		checkLevel1Pair(t, "rotm diagonal", n, gotX, gotY, wantX, wantY)
	}
}

func TestLevel1UnitarySIMDOverlap(t *testing.T) {
	tests := []struct {
		name           string
		xStart, yStart int
	}{
		{name: "same", xStart: 0, yStart: 0},
		{name: "y ahead", xStart: 0, yStart: 1},
		{name: "x ahead", xStart: 1, yStart: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const n = 20
			initial := make([]float32, n+1)
			for i := range initial {
				initial[i] = float32(i + 1)
			}

			checkOverlap := func(name string, scalar, simd func(x, y []float32)) {
				want := slices.Clone(initial)
				scalar(want[test.xStart:test.xStart+n], want[test.yStart:test.yStart+n])
				got := slices.Clone(initial)
				simd(got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n])
				if !slices.Equal(got, want) {
					t.Errorf("%s: got %v want %v", name, got, want)
				}
			}

			checkOverlap("swap", swapUnitaryScalar, SwapUnitary)
			checkOverlap("rot", func(x, y []float32) { rotUnitaryScalar(x, y, 0.5, -0.25) }, func(x, y []float32) { RotUnitary(x, y, 0.5, -0.25) })
			checkOverlap("rotm rescaling", func(x, y []float32) { rotmUnitaryRescalingScalar(x, y, 0.5, 0.25, -0.25, 0.5) }, func(x, y []float32) { RotmUnitaryRescaling(x, y, 0.5, 0.25, -0.25, 0.5) })
			checkOverlap("rotm off-diagonal", func(x, y []float32) { rotmUnitaryOffDiagonalScalar(x, y, 0.25, -0.25) }, func(x, y []float32) { RotmUnitaryOffDiagonal(x, y, 0.25, -0.25) })
			checkOverlap("rotm diagonal", func(x, y []float32) { rotmUnitaryDiagonalScalar(x, y, 0.5, 0.5) }, func(x, y []float32) { RotmUnitaryDiagonal(x, y, 0.5, 0.5) })
		})
	}
}

func checkLevel1Pair(t *testing.T, name string, n int, gotX, gotY, wantX, wantY []float32) {
	t.Helper()
	if !slices.Equal(gotX, wantX) || !slices.Equal(gotY, wantY) {
		t.Errorf("%s n=%d: got (%v, %v) want (%v, %v)", name, n, gotX, gotY, wantX, wantY)
	}
}
