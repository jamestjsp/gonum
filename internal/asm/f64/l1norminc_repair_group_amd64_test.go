// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !noasm && !safe && !gccgo

package f64

import (
	"fmt"
	"math"
	"testing"
)

// This is an independent scalar representation of the ORIGINAL Inc grouping,
// not the unitary grouping and not the max(sum+x,sum-x) implementation.
// Each logical lane gets indices congruent modulo8, rounded on every addition.
func l1IncRepairGrouped(x []float64, n, inc int) float64 {
	var lanes [8]float64
	blocks := n / 8
	for block := 0; block < blocks; block++ {
		for lane := 0; lane < 8; lane++ {
			lanes[lane] += math.Abs(x[(8*block+lane)*inc])
		}
	}
	var sum float64
	if blocks != 0 {
		low01 := lanes[0] + lanes[1]
		low32 := lanes[3] + lanes[2]
		low := low01 + low32
		high45 := lanes[4] + lanes[5]
		high76 := lanes[7] + lanes[6]
		high := high45 + high76
		sum = high + low // original SHUFPD then ADDSD: upper + lower
	}
	for k := 8 * blocks; k < n; k++ {
		sum += math.Abs(x[k*inc])
	}
	return sum
}

func l1IncRepairGroupedValues(n int, kind string) []float64 {
	values := make([]float64, n)
	for i := range values {
		var value float64
		switch kind {
		case "rounding-order":
			// Same-lane accumulation differs from serial accumulation for
			// suitable n: the exact representation check protects grouping.
			value = []float64{1 << 53, 1, 3, 1, 1, 5, 1, 7}[i%8]
		case "wide-normal":
			exponent := uint64(20 + (i*109)%1900)
			fraction := (uint64(i+1) * 0x123456789ab) & ((uint64(1) << 52) - 1)
			value = math.Float64frombits(exponent<<52 | fraction)
		case "overflow":
			value = math.MaxFloat64 / 16
		case "subnormal":
			value = math.Float64frombits(uint64(1 + (i*17)%997))
		default:
			panic("unknown grouping pattern")
		}
		if i%3 == 1 {
			value = -value
		}
		values[i] = value
	}
	return values
}

func TestL1NormIncRepairGrouping(t *testing.T) {
	lengths := []int{8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257}
	for _, n := range lengths {
		for _, inc := range l1IncRepairStrides {
			for _, kind := range []string{"rounding-order", "wide-normal", "overflow", "subnormal"} {
				t.Run(fmt.Sprintf("n=%d/inc=%d/%s", n, inc, kind), func(t *testing.T) {
					storage, x := l1IncRepairStorage(l1IncRepairGroupedValues(n, kind), inc)
					before := l1IncRepairBits(storage)
					want := l1IncRepairGrouped(x, n, inc)
					got := L1NormInc(x, n, inc)
					l1IncRepairCheck(t, storage, before, got, want)
				})
			}
		}
	}
}

// These64 controls explicitly supersede the frozen MH repeated-x[0] controls.
// R6E2OJ follows the documented empty-loop contract: inc=0 returns positive zero.
// The old MH source and observed outcomes remain unchanged historical evidence.
func TestL1NormIncRepairZeroStrideDocumented(t *testing.T) {
	values := []struct {
		name  string
		value float64
	}{
		{"positive-zero", 0}, {"negative-zero", math.Copysign(0, -1)},
		{"positive-dyadic", 2.5}, {"negative-dyadic", -2.5},
		{"positive-inf", math.Inf(1)}, {"negative-inf", math.Inf(-1)},
		{"nan", math.NaN()}, {"subnormal", math.SmallestNonzeroFloat64},
	}
	for _, n := range []int{1, 2, 7, 8, 9, 16, 17, 65} {
		for _, value := range values {
			t.Run(fmt.Sprintf("n=%d/%s", n, value.name), func(t *testing.T) {
				storage, x := l1IncRepairStorage([]float64{value.value}, 0)
				before := l1IncRepairBits(storage)
				want := float64(0)
				got := L1NormInc(x, n, 0)
				l1IncRepairCheck(t, storage, before, got, want)
			})
		}
	}
}
