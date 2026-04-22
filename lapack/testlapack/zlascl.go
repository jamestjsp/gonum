// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/lapack"
)

type Zlascler interface {
	Zlascl(kind lapack.MatrixType, kl, ku int, cfrom, cto float64, m, n int, a []complex128, lda int)
}

func ZlasclTest(t *testing.T, impl Zlascler) {
	const tol = 1e-14

	rnd := rand.New(rand.NewPCG(1, 1))
	for ti, test := range []struct {
		m, n int
	}{
		{0, 0},
		{1, 1},
		{1, 10},
		{10, 1},
		{2, 2},
		{3, 11},
		{11, 3},
		{11, 11},
	} {
		m := test.m
		n := test.n
		for _, extra := range []int{0, 7} {
			for _, kind := range []lapack.MatrixType{lapack.General, lapack.UpperTri, lapack.LowerTri} {
				lda := n + extra
				if lda == 0 {
					lda = 1
				}
				a := make([]complex128, max(1, m)*lda)
				aCopy := make([]complex128, len(a))
				for i := range a {
					a[i] = complex(rnd.NormFloat64(), rnd.NormFloat64())
					aCopy[i] = a[i]
				}
				cfrom := rnd.NormFloat64()
				cto := rnd.NormFloat64()
				scale := cto / cfrom

				impl.Zlascl(kind, -1, -1, cfrom, cto, m, n, a, lda)

				prefix := fmt.Sprintf("Case #%v: kind=%c,m=%v,n=%v,extra=%v", ti, kind, m, n, extra)

				switch kind {
				case lapack.UpperTri:
					for i := 0; i < m; i++ {
						for j := 0; j < min(i, n); j++ {
							if a[i*lda+j] != aCopy[i*lda+j] {
								t.Errorf("%v: unexpected lower-triangle write at (%d,%d)", prefix, i, j)
							}
						}
					}
				case lapack.LowerTri:
					for i := 0; i < m; i++ {
						for j := i + 1; j < n; j++ {
							if a[i*lda+j] != aCopy[i*lda+j] {
								t.Errorf("%v: unexpected upper-triangle write at (%d,%d)", prefix, i, j)
							}
						}
					}
				}

				var resid float64
				switch kind {
				case lapack.General:
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							want := complex(scale, 0) * aCopy[i*lda+j]
							got := a[i*lda+j]
							resid = math.Max(resid, cmplx.Abs(want-got))
						}
					}
				case lapack.UpperTri:
					for i := 0; i < m; i++ {
						for j := i; j < n; j++ {
							want := complex(scale, 0) * aCopy[i*lda+j]
							got := a[i*lda+j]
							resid = math.Max(resid, cmplx.Abs(want-got))
						}
					}
				case lapack.LowerTri:
					for i := 0; i < m; i++ {
						for j := 0; j <= min(i, n-1); j++ {
							want := complex(scale, 0) * aCopy[i*lda+j]
							got := a[i*lda+j]
							resid = math.Max(resid, cmplx.Abs(want-got))
						}
					}
				}
				if resid > tol*float64(max(m, n)) {
					t.Errorf("%v: residual=%v want<=%v", prefix, resid, tol*float64(max(m, n)))
				}
			}
		}
	}
}
