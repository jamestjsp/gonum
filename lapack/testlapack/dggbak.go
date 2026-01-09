// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

type Dggbaker interface {
	Dggbak(job lapack.BalanceJob, side blas.Side, n, ilo, ihi int, lscale, rscale []float64, m int, v []float64, ldv int)
}

func DggbakTest(t *testing.T, impl Dggbaker) {
	// Test basic parameter validation and simple cases.

	for _, n := range []int{0, 1, 2, 5, 10} {
		for _, job := range []lapack.BalanceJob{lapack.BalanceNone, lapack.Permute, lapack.Scale, lapack.PermuteScale} {
			for _, side := range []blas.Side{blas.Left, blas.Right} {
				testDggbak(t, impl, n, job, side)
			}
		}
	}
}

func testDggbak(t *testing.T, impl Dggbaker, n int, job lapack.BalanceJob, side blas.Side) {
	if n == 0 {
		// Test empty case.
		impl.Dggbak(job, side, 0, 0, -1, nil, nil, 0, nil, 1)
		return
	}

	m := n
	ilo, ihi := 0, n-1

	// Initialize scale arrays.
	lscale := make([]float64, n)
	rscale := make([]float64, n)
	for i := 0; i < n; i++ {
		lscale[i] = 1
		rscale[i] = 1
	}

	// Initialize eigenvector matrix as identity.
	v := make([]float64, n*m)
	for i := 0; i < n; i++ {
		v[i*m+i] = 1
	}

	// Call Dggbak.
	impl.Dggbak(job, side, n, ilo, ihi, lscale, rscale, m, v, m)

	// With scale=1 and ilo=0, ihi=n-1, V should be unchanged.
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if v[i*m+j] != expected {
				t.Errorf("n=%d, job=%c, side=%c: v[%d,%d]=%v, want %v",
					n, job, side, i, j, v[i*m+j], expected)
			}
		}
	}
}
