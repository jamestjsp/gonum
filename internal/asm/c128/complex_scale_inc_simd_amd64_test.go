// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"slices"
	"testing"
)

// A one-element span remains valid for any stride, including one whose byte
// conversion wraps. Invalid native spans must retain writes before the panic.
func TestSIMDComplexScaleSpanEdges(t *testing.T) {
	if !complexNativeSIMD() {
		t.Skip("native complex kernel unavailable")
	}
	for _, tc := range []struct {
		length int
		n, inc uintptr
	}{
		{0, 0, ^uintptr(0)},
		{0, 1, 3},
		{1, 1, ^uintptr(0)},
		{1, 1, 1 << 63},
		{1, 1, 1<<63 - 1},
		{1, 4, 0},
		{1, 2, ^uintptr(0)},
		{3, 3, 2},
		{10, 5, 3},
		{24, 9, 3},
		{65, 33, 2},
	} {
		for _, realScale := range []bool{false, true} {
			got := make([]complex128, tc.length)
			for i := range got {
				got[i] = complex(float64(i+1)/4, -float64(i+1)/8)
			}
			want := slices.Clone(got)
			panics := func(f func()) (didPanic bool) {
				defer func() { didPanic = recover() != nil }()
				f()
				return false
			}
			wantPanic := panics(func() {
				var ix uintptr
				for n := tc.n; n > 0; n-- {
					if realScale {
						v := want[ix]
						want[ix] = complex(0.5*real(v), 0.5*imag(v))
					} else {
						want[ix] *= 0.75 - 0.25i
					}
					ix += tc.inc
				}
			})
			gotPanic := panics(func() {
				if realScale {
					DscalIncSIMD(0.5, got, tc.n, tc.inc)
				} else {
					ScalIncSIMD(0.75-0.25i, got, tc.n, tc.inc)
				}
			})
			if gotPanic != wantPanic || !slices.Equal(got, want) {
				t.Errorf("len=%d n=%d inc=%d real=%t: panic=%t want %t; got %v want %v", tc.length, tc.n, tc.inc, realScale, gotPanic, wantPanic, got, want)
			}
		}
	}
}
