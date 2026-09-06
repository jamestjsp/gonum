// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "testing"

func TestRecheckDggesSelectionMisorderedPair(t *testing.T) {
	for _, imagSign := range []float64{-1, 1} {
		calls := 0
		selector := func(_, ai, _ float64) bool {
			calls++
			return ai*imagSign > 0
		}
		sdim, ok := recheckDggesSelection(true, selector,
			[]float64{1, 2, 2}, []float64{0, 1, -1}, []float64{1, 1, 1})
		if ok || sdim != 2 || calls != 3 {
			t.Fatalf("imagSign=%g: ok=%v sdim=%d calls=%d, want false,2,3", imagSign, ok, sdim, calls)
		}
	}
}

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
