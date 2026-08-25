// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "testing"

func TestRecheckDggesSelectionAfterFailure(t *testing.T) {
	calls := 0
	selector := func(alphar, _, _ float64) bool {
		calls++
		return alphar < 0
	}
	sdim, ok := recheckDggesSelection(false, selector,
		[]float64{-1, 2}, []float64{0, 0}, []float64{1, 1})
	if ok {
		t.Fatal("prior reordering failure was lost")
	}
	if calls != 2 || sdim != 1 {
		t.Fatalf("calls=%d sdim=%d, want calls=2 sdim=1", calls, sdim)
	}
}
