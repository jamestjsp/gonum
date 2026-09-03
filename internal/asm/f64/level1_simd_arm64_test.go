// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"slices"
	"testing"
)

func TestDasumUnitarySIMDLengths(t *testing.T) {
	for n := 0; n <= 33; n++ {
		x := make([]float64, n)
		var want float64
		for i := range x {
			x[i] = float64(i%7 - 3)
			if x[i] < 0 {
				want -= x[i]
			} else {
				want += x[i]
			}
		}
		if got := DasumUnitary(x); got != want {
			t.Errorf("n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestLevel1UnitarySIMDLengths(t *testing.T) {
	for n := 0; n <= 17; n++ {
		x := make([]float64, n)
		y := make([]float64, n)
		for i := range x {
			x[i] = float64(i - 4)
			y[i] = float64(2*i + 1)
		}

		wantX := slices.Clone(y)
		wantY := slices.Clone(x)
		gotX := slices.Clone(x)
		gotY := slices.Clone(y)
		SwapUnitary(gotX, gotY)
		if !slices.Equal(gotX, wantX) || !slices.Equal(gotY, wantY) {
			t.Errorf("swap n=%d: got (%v, %v) want (%v, %v)", n, gotX, gotY, wantX, wantY)
		}

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		for i, vx := range wantX {
			vy := wantY[i]
			wantX[i], wantY[i] = 2*vx+3*vy, 2*vy-3*vx
		}
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotUnitary(gotX, gotY, 2, 3)
		if !slices.Equal(gotX, wantX) || !slices.Equal(gotY, wantY) {
			t.Errorf("rot n=%d: got (%v, %v) want (%v, %v)", n, gotX, gotY, wantX, wantY)
		}

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		for i, vx := range wantX {
			vy := wantY[i]
			wantX[i], wantY[i] = 2*vx+3*vy, 4*vx+5*vy
		}
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotmUnitaryRescaling(gotX, gotY, 2, 3, 4, 5)
		if !slices.Equal(gotX, wantX) || !slices.Equal(gotY, wantY) {
			t.Errorf("rotm n=%d: got (%v, %v) want (%v, %v)", n, gotX, gotY, wantX, wantY)
		}

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		for i, vx := range wantX {
			vy := wantY[i]
			wantX[i], wantY[i] = vx+3*vy, 4*vx+vy
		}
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotmUnitaryOffDiagonal(gotX, gotY, 3, 4)
		if !slices.Equal(gotX, wantX) || !slices.Equal(gotY, wantY) {
			t.Errorf("rotm off-diagonal n=%d: got (%v, %v) want (%v, %v)", n, gotX, gotY, wantX, wantY)
		}

		wantX = slices.Clone(x)
		wantY = slices.Clone(y)
		for i, vx := range wantX {
			vy := wantY[i]
			wantX[i], wantY[i] = 2*vx+vy, -vx+5*vy
		}
		gotX = slices.Clone(x)
		gotY = slices.Clone(y)
		RotmUnitaryDiagonal(gotX, gotY, 2, 5)
		if !slices.Equal(gotX, wantX) || !slices.Equal(gotY, wantY) {
			t.Errorf("rotm diagonal n=%d: got (%v, %v) want (%v, %v)", n, gotX, gotY, wantX, wantY)
		}
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
			initial := make([]float64, n+1)
			for i := range initial {
				initial[i] = float64(i + 1)
			}

			want := slices.Clone(initial)
			wantX := want[test.xStart : test.xStart+n]
			wantY := want[test.yStart : test.yStart+n]
			swapUnitaryScalar(wantX, wantY)
			got := slices.Clone(initial)
			SwapUnitary(got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n])
			if !slices.Equal(got, want) {
				t.Errorf("swap: got %v want %v", got, want)
			}

			want = slices.Clone(initial)
			wantX = want[test.xStart : test.xStart+n]
			wantY = want[test.yStart : test.yStart+n]
			rotUnitaryScalar(wantX, wantY, 2, 3)
			got = slices.Clone(initial)
			RotUnitary(got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n], 2, 3)
			if !slices.Equal(got, want) {
				t.Errorf("rot: got %v want %v", got, want)
			}

			want = slices.Clone(initial)
			wantX = want[test.xStart : test.xStart+n]
			wantY = want[test.yStart : test.yStart+n]
			rotmUnitaryRescalingScalar(wantX, wantY, 2, 3, 4, 5)
			got = slices.Clone(initial)
			RotmUnitaryRescaling(got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n], 2, 3, 4, 5)
			if !slices.Equal(got, want) {
				t.Errorf("rotm: got %v want %v", got, want)
			}

			want = slices.Clone(initial)
			wantX = want[test.xStart : test.xStart+n]
			wantY = want[test.yStart : test.yStart+n]
			rotmUnitaryOffDiagonalScalar(wantX, wantY, 3, 4)
			got = slices.Clone(initial)
			RotmUnitaryOffDiagonal(got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n], 3, 4)
			if !slices.Equal(got, want) {
				t.Errorf("rotm off-diagonal: got %v want %v", got, want)
			}

			want = slices.Clone(initial)
			wantX = want[test.xStart : test.xStart+n]
			wantY = want[test.yStart : test.yStart+n]
			rotmUnitaryDiagonalScalar(wantX, wantY, 2, 5)
			got = slices.Clone(initial)
			RotmUnitaryDiagonal(got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n], 2, 5)
			if !slices.Equal(got, want) {
				t.Errorf("rotm diagonal: got %v want %v", got, want)
			}
		})
	}
}
