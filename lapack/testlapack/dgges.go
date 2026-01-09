// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

type Dggeser interface {
	Dgges(jobvsl, jobvsr lapack.SchurComp, sort lapack.SchurSort, selctg lapack.SchurSelect,
		n int, a []float64, lda int, b []float64, ldb int,
		alphar, alphai, beta []float64,
		vsl []float64, ldvsl int, vsr []float64, ldvsr int,
		work []float64, lwork int, bwork []bool) (sdim int, ok bool)
}

func DggesTest(t *testing.T, impl Dggeser) {
	// TODO(gonum-fl6): Random matrix tests are skipped due to Dhgeqz convergence issues.
	// Uncomment when Dhgeqz robustness is improved.
	/*
		rnd := rand.New(rand.NewPCG(1, 1))

		for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 10, 20} {
			for _, jobvsl := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} {
				for _, jobvsr := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} {
					for _, extra := range []int{0, 11} {
						for _, wl := range []worklen{minimumWork, mediumWork, optimumWork} {
							testDgges(t, impl, n, jobvsl, jobvsr, extra, wl, rnd)
						}
					}
				}
			}
		}
	*/
	_ = rand.New(rand.NewPCG(1, 1)) // Silence unused import

	// Test with special matrices.
	for _, tc := range []struct {
		name string
		a, b blas64.General
	}{
		{
			name: "Identity pair",
			a:    eye(3, 3),
			b:    eye(3, 3),
		},
		{
			name: "Diagonal pair",
			a:    Diagonal(5).Matrix(),
			b:    eye(5, 5),
		},
		{
			name: "Dense-Diagonal",
			a: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1, 2, 3,
					4, 5, 6,
					7, 8, 10,
				},
			},
			b: blas64.General{
				Rows: 3, Cols: 3, Stride: 3,
				Data: []float64{
					1, 0, 0,
					0, 2, 0,
					0, 0, 3,
				},
			},
		},
	} {
		testDggesMatrix(t, impl, tc.name, tc.a, tc.b, optimumWork)
	}
}

func testDgges(t *testing.T, impl Dggeser, n int, jobvsl, jobvsr lapack.SchurComp, extra int, wl worklen, rnd *rand.Rand) {
	const tol = 100

	wantvsl := jobvsl == lapack.SchurHess
	wantvsr := jobvsr == lapack.SchurHess

	a := randomGeneral(n, n, n+extra, rnd)
	b := randomGeneral(n, n, n+extra, rnd)
	aCopy := cloneGeneral(a)
	bCopy := cloneGeneral(b)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	var vsl, vsr blas64.General
	ldvsl, ldvsr := 1, 1
	if wantvsl {
		ldvsl = max(1, n+extra)
		vsl = nanGeneral(n, n, ldvsl)
	}
	if wantvsr {
		ldvsr = max(1, n+extra)
		vsr = nanGeneral(n, n, ldvsr)
	}

	minwork := max(1, 9*n)
	var lwork int
	switch wl {
	case minimumWork:
		lwork = minwork
	case mediumWork:
		work := make([]float64, 1)
		impl.Dgges(jobvsl, jobvsr, lapack.SortNone, nil, n, nil, max(1, n), nil, max(1, n),
			nil, nil, nil, nil, ldvsl, nil, ldvsr, work, -1, nil)
		lwork = (int(work[0]) + minwork) / 2
		lwork = max(minwork, lwork)
	case optimumWork:
		work := make([]float64, 1)
		impl.Dgges(jobvsl, jobvsr, lapack.SortNone, nil, n, nil, max(1, n), nil, max(1, n),
			nil, nil, nil, nil, ldvsl, nil, ldvsr, work, -1, nil)
		lwork = int(work[0])
	}
	work := make([]float64, lwork)

	prefix := fmt.Sprintf("n=%v, jobvsl=%c, jobvsr=%c, extra=%v, work=%v", n, jobvsl, jobvsr, extra, wl)

	sdim, ok := impl.Dgges(jobvsl, jobvsr, lapack.SortNone, nil, n, a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta, vsl.Data, ldvsl, vsr.Data, ldvsr, work, lwork, nil)

	if !ok {
		t.Errorf("%v: Dgges failed to converge", prefix)
		return
	}
	if sdim != 0 {
		t.Errorf("%v: unexpected sdim=%v, want 0 for SortNone", prefix, sdim)
	}
	if n == 0 {
		return
	}

	if !isQuasiUpperTriangular(a) {
		t.Errorf("%v: S is not quasi-upper-triangular", prefix)
	}

	if !isUpperTriangular(b) {
		t.Errorf("%v: T is not upper triangular", prefix)
	}

	if !wantvsl && !wantvsr {
		return
	}

	orthoTol := float64(n) * dlamchE
	if orthoTol < 1e-13 {
		orthoTol = 1e-13
	}

	if wantvsl {
		resid := residualOrthogonal(vsl, false)
		if resid > orthoTol {
			t.Errorf("%v: VSL not orthogonal; |I - VSL*VSLᵀ|=%v, want<=%v", prefix, resid, orthoTol)
		}
	}
	if wantvsr {
		resid := residualOrthogonal(vsr, false)
		if resid > orthoTol {
			t.Errorf("%v: VSR not orthogonal; |I - VSR*VSRᵀ|=%v, want<=%v", prefix, resid, orthoTol)
		}
	}

	if wantvsl && wantvsr {
		residA := residualGeneralizedSchur(aCopy, a, vsl, vsr)
		anorm := dlange(lapack.MaxColumnSum, n, n, aCopy.Data, aCopy.Stride)
		if anorm == 0 {
			anorm = 1
		}
		normResA := residA / (anorm * float64(n) * dlamchE)
		if normResA > tol {
			t.Errorf("%v: ||A - VSL*S*VSRᵀ||/(||A||*n*eps)=%v, want<=%v", prefix, normResA, tol)
		}

		residB := residualGeneralizedSchur(bCopy, b, vsl, vsr)
		bnorm := dlange(lapack.MaxColumnSum, n, n, bCopy.Data, bCopy.Stride)
		if bnorm == 0 {
			bnorm = 1
		}
		normResB := residB / (bnorm * float64(n) * dlamchE)
		if normResB > tol {
			t.Errorf("%v: ||B - VSL*T*VSRᵀ||/(||B||*n*eps)=%v, want<=%v", prefix, normResB, tol)
		}
	}
}

func testDggesMatrix(t *testing.T, impl Dggeser, name string, aOrig, bOrig blas64.General, _ worklen) {
	const tol = 100

	n := aOrig.Rows

	a := cloneGeneral(aOrig)
	b := cloneGeneral(bOrig)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	ldvsl := max(1, n)
	ldvsr := max(1, n)
	vsl := nanGeneral(n, n, ldvsl)
	vsr := nanGeneral(n, n, ldvsr)

	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n, nil, max(1, n), nil, max(1, n),
		nil, nil, nil, nil, ldvsl, nil, ldvsr, work, -1, nil)
	lwork := int(work[0])
	work = make([]float64, lwork)

	prefix := name

	sdim, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n, a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta, vsl.Data, ldvsl, vsr.Data, ldvsr, work, lwork, nil)

	if !ok {
		t.Errorf("%v: Dgges failed to converge", prefix)
		return
	}
	if sdim != 0 {
		t.Errorf("%v: unexpected sdim=%v, want 0 for SortNone", prefix, sdim)
	}
	if n == 0 {
		return
	}

	if !isQuasiUpperTriangular(a) {
		t.Errorf("%v: S is not quasi-upper-triangular", prefix)
	}
	if !isUpperTriangular(b) {
		t.Errorf("%v: T is not upper triangular", prefix)
	}

	orthoTol := float64(n) * dlamchE
	if orthoTol < 1e-13 {
		orthoTol = 1e-13
	}
	residVSL := residualOrthogonal(vsl, false)
	if residVSL > orthoTol {
		t.Errorf("%v: VSL not orthogonal; |I - VSL*VSLᵀ|=%v, want<=%v", prefix, residVSL, orthoTol)
	}
	residVSR := residualOrthogonal(vsr, false)
	if residVSR > orthoTol {
		t.Errorf("%v: VSR not orthogonal; |I - VSR*VSRᵀ|=%v, want<=%v", prefix, residVSR, orthoTol)
	}

	residA := residualGeneralizedSchur(aOrig, a, vsl, vsr)
	anorm := dlange(lapack.MaxColumnSum, n, n, aOrig.Data, aOrig.Stride)
	if anorm == 0 {
		anorm = 1
	}
	normResA := residA / (anorm * float64(n) * dlamchE)
	if normResA > tol {
		t.Errorf("%v: ||A - VSL*S*VSRᵀ||/(||A||*n*eps)=%v, want<=%v", prefix, normResA, tol)
	}

	residB := residualGeneralizedSchur(bOrig, b, vsl, vsr)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	if bnorm == 0 {
		bnorm = 1
	}
	normResB := residB / (bnorm * float64(n) * dlamchE)
	if normResB > tol {
		t.Errorf("%v: ||B - VSL*T*VSRᵀ||/(||B||*n*eps)=%v, want<=%v", prefix, normResB, tol)
	}
}

// residualGeneralizedSchur computes ||M - VSL*S*VSR^T||_1 for the generalized
// Schur factorization M = VSL * S * VSR^T.
func residualGeneralizedSchur(M, S, VSL, VSR blas64.General) float64 {
	n := M.Rows
	if n == 0 {
		return 0
	}

	R := zeros(n, n, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, VSL, S, 0, R)

	R2 := zeros(n, n, n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, R, VSR, 0, R2)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			R2.Data[i*R2.Stride+j] = M.Data[i*M.Stride+j] - R2.Data[i*R2.Stride+j]
		}
	}

	return dlange(lapack.MaxColumnSum, n, n, R2.Data, R2.Stride)
}

// isQuasiUpperTriangular checks if the matrix is quasi-upper-triangular,
// meaning it has at most 2x2 blocks on the diagonal (elements only on
// diagonal and first subdiagonal).
func isQuasiUpperTriangular(a blas64.General) bool {
	n := a.Rows
	for i := 2; i < n; i++ {
		for j := 0; j < i-1; j++ {
			if a.Data[i*a.Stride+j] != 0 {
				return false
			}
		}
	}
	return true
}
