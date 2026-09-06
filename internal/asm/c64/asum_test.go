// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c64

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestAsumUnitary(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for n := 0; n <= 65; n++ {
		x := make([]complex64, n)
		var want float32
		for i := range x {
			x[i] = complex(float32(rnd.NormFloat64()), float32(rnd.NormFloat64()))
			want += float32(math.Abs(float64(real(x[i]))) + math.Abs(float64(imag(x[i]))))
		}
		got := AsumUnitary(x)
		if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.IsNaN(float64(want)) || math.IsInf(float64(want), 0) || math.Abs(float64(got-want)) > 2e-6*math.Max(1, float64(want)) {
			t.Errorf("n=%d: got %g, want %g", n, got, want)
		}
	}
	inf, nan := float32(math.Inf(1)), float32(math.NaN())
	if got := AsumUnitary([]complex64{complex(float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1)))}); got != 0 || math.Signbit(float64(got)) {
		t.Errorf("signed zero: got %g, want +0", got)
	}
	long33 := make([]complex64, 33)
	long33[8], long33[32] = complex(inf, 0), complex(0, nan)
	long65 := make([]complex64, 65)
	long65[16], long65[64] = complex(nan, 0), complex(0, inf)
	for _, x := range [][]complex64{{complex(inf, 1)}, {complex(inf, 0), complex(nan, 0)}, long33, long65} {
		got := AsumUnitary(x)
		var want float32
		for _, v := range x {
			want += float32(math.Abs(float64(real(v))) + math.Abs(float64(imag(v))))
		}
		if !(math.IsNaN(float64(got)) && math.IsNaN(float64(want))) && got != want {
			t.Errorf("extreme: got %g, want %g", got, want)
		}
	}
}
