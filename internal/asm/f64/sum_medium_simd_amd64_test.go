// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"testing"
)

func TestSIMDSumMediumGroupingRecovery(t *testing.T) {
	x := make([]float64, 64)
	m := .75 * math.MaxFloat64
	x[0], x[8], x[16] = m, -m, m
	got, want := SumSIMD(x), sumOriginalSIMD(x)
	if !(got == want || math.IsNaN(got) && math.IsNaN(want)) {
		t.Fatalf("got=%g want=%g", got, want)
	}
	if !simd.Emulated() && simd.VectorBitSize() == 512 && got != m {
		t.Fatalf("lost original finite grouping: %g", got)
	}
}
