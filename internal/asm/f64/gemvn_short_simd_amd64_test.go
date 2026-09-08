// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"testing"
)

func TestGemvNShortNativeGroupingRecovery(t *testing.T) {
	const m, n, lda = 12, 8, 11
	a, x, y := make([]float64, (m-1)*lda+n), make([]float64, n), make([]float64, m)
	for i := range x {
		x[i] = 1
	}
	for row := 0; row < m; row++ {
		a[row*lda] = 2
		if row >= 4 {
			a[row*lda] = math.MaxFloat64
			a[row*lda+1] = -math.MaxFloat64
			a[row*lda+4] = math.MaxFloat64
		}
		y[row] = 1
	}
	// The first block is already stored when the second block overflows in
	// the native grouping. Recovery must neither repeat beta nor skip rows.
	GemvNSIMD(m, n, 0.5, a, lda, x, 1, 0.25, y, 1)
	for row, got := range y {
		sum := dotUnitaryOriginalSIMD(x, a[row*lda:row*lda+n])
		want := 0.5*sum + 0.25
		if got != want {
			t.Fatalf("row=%d got=%g want=%g", row, got, want)
		}
	}
}
