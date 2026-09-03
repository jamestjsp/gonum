// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f32

import (
	"slices"
	"testing"
)

func TestSIMDUnitaryTails(t *testing.T) {
	for n := 0; n <= 67; n++ {
		x := make([]float32, n)
		y := make([]float32, n)
		for i := range x {
			x[i] = float32(i%7 - 3)
			y[i] = float32(i%5 - 2)
		}

		var wantDot float32
		var wantDdot float64
		var wantSum float32
		for i, v := range x {
			wantDot += v * y[i]
			wantDdot += float64(v) * float64(y[i])
			wantSum += v
		}
		if got := DotUnitary(x, y); got != wantDot {
			t.Errorf("DotUnitary n=%d: got %v want %v", n, got, wantDot)
		}
		if got := DdotUnitary(x, y); got != wantDdot {
			t.Errorf("DdotUnitary n=%d: got %v want %v", n, got, wantDdot)
		}
		if got := Sum(x); got != wantSum {
			t.Errorf("Sum n=%d: got %v want %v", n, got, wantSum)
		}

		gotAxpy := slices.Clone(y)
		wantAxpy := slices.Clone(y)
		for i, v := range x {
			wantAxpy[i] += 2 * v
		}
		AxpyUnitary(2, x, gotAxpy)
		if !slices.Equal(gotAxpy, wantAxpy) {
			t.Errorf("AxpyUnitary n=%d: got %v want %v", n, gotAxpy, wantAxpy)
		}

		gotTo := make([]float32, n)
		AxpyUnitaryTo(gotTo, 2, x, y)
		if !slices.Equal(gotTo, wantAxpy) {
			t.Errorf("AxpyUnitaryTo n=%d: got %v want %v", n, gotTo, wantAxpy)
		}

		gotScal := slices.Clone(x)
		wantScal := slices.Clone(x)
		for i := range wantScal {
			wantScal[i] *= -3
		}
		ScalUnitary(-3, gotScal)
		if !slices.Equal(gotScal, wantScal) {
			t.Errorf("ScalUnitary n=%d: got %v want %v", n, gotScal, wantScal)
		}
		gotScalTo := make([]float32, n)
		ScalUnitaryTo(gotScalTo, -3, x)
		if !slices.Equal(gotScalTo, wantScal) {
			t.Errorf("ScalUnitaryTo n=%d: got %v want %v", n, gotScalTo, wantScal)
		}
	}
}

func TestSIMDUnitaryOverlap(t *testing.T) {
	tests := []struct {
		name      string
		dst, x, y int
		length    int
	}{
		{name: "same", dst: 1, x: 1, y: 1, length: 16},
		{name: "destination ahead", dst: 1, x: 0, y: 2, length: 16},
		{name: "destination behind", dst: 0, x: 1, y: 2, length: 16},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := make([]float32, test.length+3)
			for i := range got {
				got[i] = float32(i + 1)
			}
			want := slices.Clone(got)
			wantDst := want[test.dst : test.dst+test.length]
			wantX := want[test.x : test.x+test.length]
			wantY := want[test.y : test.y+test.length]
			axpyUnitaryToScalar(wantDst, 2, wantX, wantY)

			AxpyUnitaryTo(got[test.dst:test.dst+test.length], 2, got[test.x:test.x+test.length], got[test.y:test.y+test.length])
			if !slices.Equal(got, want) {
				t.Fatalf("got %v want %v", got, want)
			}
		})
	}
}

func TestSIMDScalUnitaryToOverlap(t *testing.T) {
	for _, offset := range []int{-1, 0, 1} {
		got := make([]float32, 18)
		for i := range got {
			got[i] = float32(i + 1)
		}
		want := slices.Clone(got)
		xStart := 1
		dstStart := xStart + offset
		scalUnitaryToScalar(want[dstStart:dstStart+16], 2, want[xStart:xStart+16])

		ScalUnitaryTo(got[dstStart:dstStart+16], 2, got[xStart:xStart+16])
		if !slices.Equal(got, want) {
			t.Errorf("offset=%d: got %v want %v", offset, got, want)
		}
	}
}
