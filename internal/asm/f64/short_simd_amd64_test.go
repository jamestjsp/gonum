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

func TestSIMDShortReductionGroupingRecovery(t *testing.T) {
	m := 0.75 * math.MaxFloat64
	x, ones := make([]float64, 10), make([]float64, 10)
	x[0], x[1], x[8] = m, -m, m
	for i := range ones {
		ones[i] = 1
	}
	// The short helper folds the last pair into its first accumulator,
	// overflowing lane zero before the lanes cancel. The established 256-
	// and 512-bit paths instead reduce their full vectors before these two
	// tail values, returning m. Recover the established grouping, including
	// its non-finite result at widths where that grouping also overflows.
	for _, test := range []struct {
		name      string
		got, want float64
	}{
		{"Sum", SumSIMD(x), sumOriginalSIMD(x)},
		{"DotUnitary", DotUnitarySIMD(x, ones), dotUnitaryOriginalSIMD(x, ones)},
	} {
		if width := simd.VectorBitSize(); (width == 256 || width == 512) && test.want != m {
			t.Fatalf("%s: original %d-bit grouping got %g want %g", test.name, width, test.want, m)
		}
		if test.got != test.want && !(math.IsNaN(test.got) && math.IsNaN(test.want)) {
			t.Errorf("%s: got %g want original grouping %g", test.name, test.got, test.want)
		}
	}
}
