// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import (
	"fmt"
	"math"
	"testing"
)

func r6CheckZeroStrideRepair(t *testing.T, fn func([]float64, int, int) float64) {
	t.Helper()
	maxInt := int(^uint(0) >> 1)
	emptyBacking := []float64{math.Float64frombits(0x7ff8000000000021)}
	inputs := []struct {
		name string
		x    []float64
	}{
		{"nil", nil},
		{"empty", []float64{}},
		{"empty-capacity", emptyBacking[:0:1]},
		{"positive-zero", []float64{0}},
		{"negative-zero", []float64{math.Copysign(0, -1)}},
		{"negative-finite", []float64{-3}},
		{"positive-inf", []float64{math.Inf(1)}},
		{"negative-inf", []float64{math.Inf(-1)}},
		{"quiet-nan", []float64{math.Float64frombits(0xfff8000000000021)}},
		{"signaling-nan-bits", []float64{math.Float64frombits(0x7ff0000000000001)}},
		{"subnormal", []float64{math.SmallestNonzeroFloat64}},
		{"maximum-finite", []float64{math.MaxFloat64}},
	}
	for _, n := range []int{0, 1, 2, 7, 8, 9, 16, 17, 65, maxInt} {
		for _, input := range inputs {
			t.Run(fmt.Sprintf("zero/n=%d/%s", n, input.name), func(t *testing.T) {
				// Inspect the real backing store too, including empty-capacity.
				backing := input.x[:cap(input.x)]
				before := make([]uint64, len(backing))
				for i, value := range backing {
					before[i] = math.Float64bits(value)
				}
				if got := fn(input.x, n, 0); math.Float64bits(got) != 0 {
					t.Fatalf("zero stride must return +0, got %g (%016x)", got, math.Float64bits(got))
				}
				for i, value := range backing {
					if math.Float64bits(value) != before[i] {
						t.Fatalf("input/backing changed at %d", i)
					}
				}
			})
		}
	}
	for _, inc := range []int{-maxInt - 1, -1, 1, 2, 7, maxInt} {
		t.Run(fmt.Sprintf("zero-count/inc=%d", inc), func(t *testing.T) {
			if got := fn(nil, 0, inc); math.Float64bits(got) != 0 {
				t.Fatalf("zero count must return +0, got %g", got)
			}
		})
	}
	// For n=1 these are valid bounded reads. In ASM the two large powers
	// produce a zero byte stride after SHLQ; only the unscaled inc==0 is empty.
	for _, inc := range []int{1, 2, 7, maxInt/4 + 1, maxInt/2 + 1, maxInt} {
		t.Run(fmt.Sprintf("positive-single/inc=%d", inc), func(t *testing.T) {
			x := []float64{-3}
			if got := fn(x, 1, inc); math.Float64bits(got) != math.Float64bits(3) {
				t.Fatalf("positive one-element stride must read x[0], got %g", got)
			}
			if math.Float64bits(x[0]) != math.Float64bits(-3) {
				t.Fatal("positive input changed")
			}
		})
	}
}

func TestR6L1NormIncZeroNoRead(t *testing.T) {
	r6CheckZeroStrideRepair(t, L1NormInc)
}
