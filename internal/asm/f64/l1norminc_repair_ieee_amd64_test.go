// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64

package f64

import (
	"fmt"
	"math"
	"testing"
)

var l1IncRepairLengths = [...]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}
var l1IncRepairStrides = [...]int{1, 2, 3, 7, 16}
var l1IncRepairPatterns = [...]string{"dyadic", "positive-zero", "negative-zero", "alternating-zero", "positive-inf", "negative-inf", "alternating-inf", "nan-first", "nan-last", "nan-inf", "overflow", "subnormal"}

// All calls using this fixture have bounded n and nonnegative increments.
// Positive increments use the minimal legal span, with NaN gaps and two outer
// sentinels at each end. No invalid slice, forged header or retry is used.
func l1IncRepairStorage(values []float64, inc int) (storage, x []float64) {
	span := 1
	if len(values) > 0 && inc > 0 {
		span = 1 + (len(values)-1)*inc
	}
	storage = make([]float64, span+4)
	for i := range storage {
		storage[i] = math.Float64frombits(0x7ff8000000004000 + uint64(i))
	}
	x = storage[2 : 2+span : 2+span]
	for k, v := range values {
		x[k*inc] = v
	}
	return storage, x
}

func l1IncRepairBits(x []float64) []uint64 {
	bits := make([]uint64, len(x))
	for i, v := range x {
		bits[i] = math.Float64bits(v)
	}
	return bits
}

func l1IncRepairCheck(t *testing.T, storage []float64, before []uint64, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("expected NaN, got %g", got)
		}
	} else if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("got %g (%016x), want %g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
	if len(storage) != len(before) {
		t.Fatal("storage length changed")
	}
	for i, v := range storage {
		if math.Float64bits(v) != before[i] {
			t.Fatalf("input/gap/sentinel changed at %d", i)
		}
	}
}

// Count logical elements explicitly. For inc>0 this is independent of the
// implementation's n*inc termination and packed-eight-lane reduction.
// Zero-stride contracts are checked separately without this positive-stride oracle.
func l1IncRepairScalar(x []float64, n, inc int) (sum float64) {
	for k := 0; k < n; k++ {
		sum += math.Abs(x[k*inc])
	}
	return sum
}

func l1IncRepairValues(n int, pattern string) []float64 {
	x := make([]float64, n)
	for k := range x {
		v := float64(1+(k*13+5)%31) / 64
		if k%2 != 0 {
			v = -v
		}
		switch pattern {
		case "dyadic":
		case "positive-zero":
			v = 0
		case "negative-zero":
			v = math.Copysign(0, -1)
		case "alternating-zero":
			v = 0
			if k%2 != 0 {
				v = math.Copysign(0, -1)
			}
		case "positive-inf":
			v = math.Inf(1)
		case "negative-inf":
			v = math.Inf(-1)
		case "alternating-inf":
			v = math.Inf(1)
			if k%2 != 0 {
				v = math.Inf(-1)
			}
		case "nan-first", "nan-last", "nan-inf":
		case "overflow":
			v = math.MaxFloat64
			if k%2 != 0 {
				v = -v
			}
		case "subnormal":
			v = math.Float64frombits(uint64(1 + k%7))
			if k%2 != 0 {
				v = -v
			}
		default:
			panic("unknown fixed IEEE pattern")
		}
		x[k] = v
	}
	switch pattern {
	case "nan-first":
		x[0] = math.Float64frombits(0xfff8000000000021)
	case "nan-last":
		x[n-1] = math.Float64frombits(0x7ff8000000000023)
	case "nan-inf":
		x[0] = math.NaN()
		if n > 1 {
			x[n-1] = math.Inf(1)
		}
	}
	return x
}

func TestL1NormIncRepairIEEE(t *testing.T) {
	for _, n := range l1IncRepairLengths {
		for _, inc := range l1IncRepairStrides {
			for _, pattern := range l1IncRepairPatterns {
				t.Run(fmt.Sprintf("n=%d/inc=%d/%s", n, inc, pattern), func(t *testing.T) {
					values := l1IncRepairValues(n, pattern)
					storage, x := l1IncRepairStorage(values, inc)
					before := l1IncRepairBits(storage)
					want := l1IncRepairScalar(x, n, inc)
					got := L1NormInc(x, n, inc)
					l1IncRepairCheck(t, storage, before, got, want)
				})
			}
		}
	}
}

func TestL1NormIncRepairZeroCount(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, inc := range []int{-maxInt - 1, -1, 0, 1, 2, maxInt} {
		t.Run(fmt.Sprintf("inc=%d", inc), func(t *testing.T) {
			got := L1NormInc(nil, 0, inc)
			if math.Float64bits(got) != 0 {
				t.Fatalf("n=0 must return +0, got %g", got)
			}
		})
	}
}
