// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import (
	"math"
	"testing"
)

// This asserts the existing documented loop, not a new broadcast contract.
// Original ASM is expected to reveal a named failure; no runtime outcome is
// fabricated by this source-only witness.
func TestR6L1NormIncZeroStride(t *testing.T) {
	x := []float64{-3}
	got := L1NormInc(x, 2, 0)
	t.Logf("R6_ZERO route=L1NormInc n=2 inc=0 len=%d cap=%d got_bits=%016x documented_bits=%016x", len(x), cap(x), math.Float64bits(got), uint64(0))
	if math.Float64bits(x[0]) != math.Float64bits(-3) {
		t.Fatal("input changed")
	}
	if math.Float64bits(got) != 0 {
		t.Fatalf("documented zero-stride loop returns +0; got %g", got)
	}
}
