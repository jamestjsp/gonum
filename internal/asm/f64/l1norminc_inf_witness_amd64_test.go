// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package f64

import (
	"math"
	"testing"
)

// This is an ordinary regression assertion, not an expected-failure switch.
// Root must run it first against the untouched original and retain the actual
// result. Source inspection predicts the original ASM fails; none is claimed.
func TestL1NormIncPositiveInfWitness(t *testing.T) {
	poison := math.Float64frombits(0x7ff8000000000051)
	storage := []float64{poison, math.Inf(1), poison, math.Inf(1), poison}
	x := storage[1:4:4]
	before := make([]uint64, len(storage))
	for i, v := range storage {
		before[i] = math.Float64bits(v)
	}
	var want float64
	for k := 0; k < 2; k++ {
		want += math.Abs(x[k*2])
	}
	got := L1NormInc(x, 2, 2)
	t.Logf("L1_INC_WITNESS n=2 inc=2 len=%d cap=%d got=%g got_bits=%016x want=%g want_bits=%016x", len(x), cap(x), got, math.Float64bits(got), want, math.Float64bits(want))
	for i, v := range storage {
		if math.Float64bits(v) != before[i] {
			t.Fatalf("input/sentinel changed at %d", i)
		}
	}
	if !math.IsInf(want, 1) {
		t.Fatal("independent witness oracle is not +Inf")
	}
	if !math.IsInf(got, 1) {
		t.Fatalf("L1NormInc must return +Inf for two positive infinities: got %g", got)
	}
}
