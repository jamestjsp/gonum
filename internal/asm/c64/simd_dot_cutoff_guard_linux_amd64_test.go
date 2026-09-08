// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import "testing"

func TestSIMDComplexDotCutoffGuardPages(t *testing.T) {
	for _, n := range []int{127, 128, 129, 255, 256, 257, 511, 512} {
		for _, xAtEnd := range []bool{false, true} {
			x := complexGuardedSliceSIMD(t, n, xAtEnd)
			y := complexGuardedSliceSIMD(t, n, !xAtEnd)
			var wantc, wantu complex64
			for i := range x {
				x[i] = complex(float32(i%13-6)/16, float32(i%7-3)/8)
				y[i] = complex(float32(i%11-5)/8, float32(i%17-8)/16)
				wantc += conj64(x[i]) * y[i]
				wantu += x[i] * y[i]
			}
			if got := DotcUnitarySIMD(x, y); got != wantc {
				t.Errorf("Dotc n=%d xAtEnd=%t got %v want %v", n, xAtEnd, got, wantc)
			}
			if got := DotuUnitarySIMD(x, y); got != wantu {
				t.Errorf("Dotu n=%d xAtEnd=%t got %v want %v", n, xAtEnd, got, wantu)
			}
		}
	}
}
