// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo
// +build go1.27,goexperiment.simd,arm64,!safe,!noasm,!gccgo

package f64_test

import (
	"testing"

	. "gonum.org/v1/gonum/internal/asm/f64"
)

func TestDotUnitarySIMDEdgeLengths(t *testing.T) {
	for n := 0; n <= 17; n++ {
		x := make([]float64, n)
		y := make([]float64, n)
		var want float64
		for i := range x {
			x[i] = float64(i%5 - 2)
			y[i] = float64(i%7 - 3)
			want += x[i] * y[i]
		}

		if got := DotUnitary(x, y); got != want {
			t.Errorf("n=%d: unexpected result: got %v want %v", n, got, want)
		}
	}
}
