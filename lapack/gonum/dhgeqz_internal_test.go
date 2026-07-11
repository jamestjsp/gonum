// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"testing"
)

func TestDlarfg3(t *testing.T) {
	tests := []struct {
		alpha float64
		x1    float64
		x2    float64
	}{
		{alpha: 3, x1: 4},
		{alpha: -3, x1: 4, x2: 5},
		{alpha: 1},
		{alpha: math.SmallestNonzeroFloat64, x1: -math.SmallestNonzeroFloat64, x2: math.SmallestNonzeroFloat64},
		{alpha: math.MaxFloat64 / 4, x1: -math.MaxFloat64 / 8, x2: math.MaxFloat64 / 16},
	}

	impl := Implementation{}
	for _, test := range tests {
		x := []float64{test.x1, test.x2}
		wantBeta, wantTau := impl.Dlarfg(3, test.alpha, x, 1)
		gotBeta, gotTau, gotX1, gotX2 := impl.dlarfg3(test.alpha, test.x1, test.x2)
		for name, pair := range map[string][2]float64{
			"beta": {gotBeta, wantBeta},
			"tau":  {gotTau, wantTau},
			"x1":   {gotX1, x[0]},
			"x2":   {gotX2, x[1]},
		} {
			if !closeDlarfg3(pair[0], pair[1]) {
				t.Errorf("alpha=%g x1=%g x2=%g: %s=%g, want %g", test.alpha, test.x1, test.x2, name, pair[0], pair[1])
			}
		}
	}
}

func closeDlarfg3(got, want float64) bool {
	if got == want {
		return true
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	return math.Abs(got-want) <= 4*dlamchP*scale
}
