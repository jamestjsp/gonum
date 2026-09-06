// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"
)

func TestScnrm2MagnitudeBoundaries(t *testing.T) {
	scales := []float32{
		math.SmallestNonzeroFloat32,
		1e-40,
		1e-15,
		1,
		1e15,
		1e30,
	}
	for _, n := range []int{0, 1, 4, 31, 32, 33, 65, 257} {
		for _, inc := range []int{1, 2} {
			for _, scale := range scales {
				t.Run(fmt.Sprintf("n=%d/inc=%d/scale=%g", n, inc, scale), func(t *testing.T) {
					x := make([]complex64, max(0, (n-1)*inc+1))
					for i := range x {
						x[i] = complex(float32(math.NaN()), float32(math.NaN()))
					}
					var sum float64
					for i := 0; i < n; i++ {
						v := complex(
							scale*float32(0.25+math.Sin(float64(i))),
							scale*float32(0.5+math.Cos(float64(i))),
						)
						x[i*inc] = v
						sum += float64(real(v))*float64(real(v)) + float64(imag(v))*float64(imag(v))
					}
					want := float32(math.Sqrt(sum))
					got := (Implementation{}).Scnrm2(n, x, inc)
					if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got-want)) > 3e-6*float64(want)+2*float64(math.SmallestNonzeroFloat32) {
						t.Fatalf("got %g, want %g", got, want)
					}
				})
			}
		}
	}
}

func TestIsamaxUnitaryOrder(t *testing.T) {
	for _, test := range []struct {
		name string
		x    []float32
		want int
	}{
		{name: "FirstNaN", x: []float32{float32(math.NaN()), 2, 3, 4, 5, 6}, want: 0},
		{name: "LaterNaN", x: []float32{1, float32(math.NaN()), 3, 4, 5, 6}, want: 5},
		{name: "FirstTie", x: []float32{1, -7, 3, 7, 5, 6}, want: 1},
		{name: "ChunkBoundary", x: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, want: 8},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := impl.Isamax(len(test.x), test.x, 1); got != test.want {
				t.Fatalf("unexpected index: got %d want %d", got, test.want)
			}
		})
	}
}
