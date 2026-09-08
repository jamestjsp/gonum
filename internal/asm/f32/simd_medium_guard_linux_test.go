// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"runtime/debug"
	"testing"
)

// Repeated vector blocks and the medium-entry cutoff retain the same exact
// input boundary as short tails, including inputs spanning several pages.
func TestSIMDGuardedReductionBoundaries(t *testing.T) {
	restore := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(restore)
	for _, n := range []int{95, 96, 97, 127, 128, 129, 255, 256, 257, 4097} {
		x, y := guardedSIMDF32(t, n), guardedSIMDF32(t, n)
		for i := range x {
			x[i], y[i] = 1, 2
		}
		if got := DotUnitarySIMD(x, y); got != float32(2*n) {
			t.Fatalf("Dot n=%d: got %g, want %g", n, got, float32(2*n))
		}
		if got := DdotUnitarySIMD(x, y); got != float64(2*n) {
			t.Fatalf("Ddot n=%d: got %g, want %g", n, got, float64(2*n))
		}
		if got := SumSIMD(x); got != float32(n) {
			t.Fatalf("Sum n=%d: got %g, want %g", n, got, float32(n))
		}
	}
}
