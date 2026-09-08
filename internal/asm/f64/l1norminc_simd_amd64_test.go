// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestSIMDL1NormIncFiniteGroupingRecovery(t *testing.T) {
	const n, inc = 32, 2
	x := make([]float64, (n-1)*inc+1)
	for i := 0; i < n; i++ {
		x[i*inc] = 0x1p967
		if i < 4 {
			x[i*inc] = math.MaxFloat64 / 4
		}
	}
	// The established four accumulators round every tiny term away before
	// combining to MaxFloat64. Eight accumulators preserve tiny-only lanes,
	// overflowing their final reduction. Retry the established grouping when
	// the faster grouping changes this finite result to infinity.
	if got := L1NormIncSIMD(x, n, inc); got != math.MaxFloat64 {
		t.Fatalf("L1NormInc: got %g want %g", got, math.MaxFloat64)
	}
}
