// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"slices"
	"testing"
)

func TestAxpyUnitarySIMDTails(t *testing.T) {
	for n := 0; n <= 17; n++ {
		x := make([]float64, n)
		y := make([]float64, n)
		want := make([]float64, n)
		for i := range x {
			x[i] = float64(i - 4)
			y[i] = float64(2*i + 1)
			want[i] = y[i] + 0.5*x[i]
		}

		AxpyUnitary(0.5, x, y)
		if !slices.Equal(y, want) {
			t.Errorf("n=%d: unexpected result: got %v want %v", n, y, want)
		}
	}
}

func TestAxpyUnitarySIMDOverlap(t *testing.T) {
	tests := []struct {
		name   string
		xStart int
		yStart int
		length int
	}{
		{name: "same", xStart: 1, yStart: 1, length: 12},
		{name: "destination ahead", xStart: 0, yStart: 1, length: 12},
		{name: "destination behind", xStart: 1, yStart: 0, length: 12},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := make([]float64, test.length+2)
			for i := range got {
				got[i] = float64(i + 1)
			}
			want := slices.Clone(got)
			wantX := want[test.xStart : test.xStart+test.length]
			wantY := want[test.yStart : test.yStart+test.length]
			for i, v := range wantX {
				wantY[i] += 2 * v
			}

			x := got[test.xStart : test.xStart+test.length]
			y := got[test.yStart : test.yStart+test.length]
			AxpyUnitary(2, x, y)
			if !slices.Equal(got, want) {
				t.Fatalf("unexpected backing slice: got %v want %v", got, want)
			}
		})
	}
}
