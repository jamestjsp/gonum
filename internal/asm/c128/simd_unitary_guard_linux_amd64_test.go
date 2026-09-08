// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"slices"
	"testing"
)

func TestSIMDComplexUnitaryNativeTailsGuardPages(t *testing.T) {
	for n := 0; n <= 65; n++ {
		x := complexGuardedSliceSIMD(t, n, true)
		y := complexGuardedSliceSIMD(t, n, false)
		dst := complexGuardedSliceSIMD(t, n, true)
		want := make([]complex128, n)
		var wantc, wantu complex128
		for i := range x {
			x[i], y[i] = complex(float64(i%7+1), -0.5), 2+0.25i
			want[i] = (0.25-0.5i)*x[i] + y[i]
			wantc += conj128(x[i]) * y[i]
			wantu += x[i] * y[i]
		}
		if got := DotcUnitarySIMD(x, y); got != wantc {
			t.Fatalf("Dotc n=%d got %v want %v", n, got, wantc)
		}
		if got := DotuUnitarySIMD(x, y); got != wantu {
			t.Fatalf("Dotu n=%d got %v want %v", n, got, wantu)
		}
		AxpyUnitaryToSIMD(dst, 0.25-0.5i, x, y)
		if !slices.Equal(dst, want) {
			t.Fatalf("AxpyTo n=%d changed values", n)
		}
		AxpyUnitarySIMD(0.25-0.5i, x, y)
		if !slices.Equal(y, want) {
			t.Fatalf("Axpy n=%d changed values", n)
		}
		for i := range x {
			want[i] = (0.25 - 0.5i) * x[i]
		}
		ScalUnitarySIMD(0.25-0.5i, x)
		if !slices.Equal(x, want) {
			t.Fatalf("Scal n=%d changed values", n)
		}
		for i, v := range x {
			want[i] = complex(0.5*real(v), 0.5*imag(v))
		}
		DscalUnitarySIMD(0.5, x)
		if !slices.Equal(x, want) {
			t.Fatalf("Dscal n=%d changed values", n)
		}

	}
}
