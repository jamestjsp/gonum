// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"fmt"
	"testing"
)

// Existing tiny IEEE tests cover offset views and n0 nonnil slices. This
// adds Dscal at each protected page boundary, including the n4 control.
func TestSIMDComplexDscalTinyBothGuardEnds(t *testing.T) {
	for n := 0; n <= 4; n++ {
		for _, atEnd := range []bool{false, true} {
			t.Run(fmt.Sprintf("n=%d/end=%t", n, atEnd), func(t *testing.T) {
				x := complexGuardedSliceSIMD(t, n, atEnd)
				for i := range x {
					x[i] = complex(float64(i+1), -0.5*float64(i+1))
				}
				DscalUnitarySIMD(-2, x)
				for i, got := range x {
					want := complex(-2*float64(i+1), float64(i+1))
					if got != want {
						t.Fatalf("index=%d got=%v want=%v", i, got, want)
					}
				}
			})
		}
	}
}
