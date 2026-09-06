// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c128

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestAsumUnitary(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for n := 0; n <= 65; n++ {
		x := make([]complex128, n)
		want := 0.0
		for i := range x {
			x[i] = complex(rnd.NormFloat64(), rnd.NormFloat64())
			want += math.Abs(real(x[i])) + math.Abs(imag(x[i]))
		}
		got := AsumUnitary(x)
		if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) || math.Abs(got-want) > 2e-15*math.Max(1, want) {
			t.Errorf("n=%d: got %g, want %g", n, got, want)
		}
	}
	if got := AsumUnitary([]complex128{complex(math.Copysign(0, -1), math.Copysign(0, -1))}); got != 0 || math.Signbit(got) {
		t.Errorf("signed zero: got %g, want +0", got)
	}
	long33 := make([]complex128, 33)
	long33[8], long33[32] = complex(math.Inf(1), 0), complex(0, math.NaN())
	long65 := make([]complex128, 65)
	long65[16], long65[64] = complex(math.NaN(), 0), complex(0, math.Inf(1))
	for _, x := range [][]complex128{{complex(math.Inf(1), 1)}, {complex(math.Inf(1), 0), complex(math.NaN(), 0)}, long33, long65} {
		got, want := AsumUnitary(x), 0.0
		for _, v := range x {
			want += math.Abs(real(v)) + math.Abs(imag(v))
		}
		if !(math.IsNaN(got) && math.IsNaN(want)) && got != want {
			t.Errorf("extreme: got %g, want %g", got, want)
		}
	}
}
