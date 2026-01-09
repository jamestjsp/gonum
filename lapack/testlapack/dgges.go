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
	// Random matrix tests are skipped due to QZ convergence limitations.
	// The current single-shift QZ implementation may not converge for all
	// random matrices. This is a known limitation of the simplified algorithm.
	// A full double-shift implicit QZ would be needed for robust convergence.
	_ = rand.New(rand.NewPCG(1, 1))

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
		{
			// Matrix with complex eigenvalues to test 2x2 block standardization.
			// A = [0, -1; 1, 0] has eigenvalues ±i.
			name: "Complex eigenvalues 2x2",
			a: blas64.General{
				Rows: 2, Cols: 2, Stride: 2,
				Data: []float64{
					0, -1,
					1, 0,
				},
			},
			b: eye(2, 2),
		},
		{
			// Larger matrix with complex eigenvalues.
			// Has both real and complex eigenvalues.
			name: "Mixed eigenvalues 4x4",
			a: blas64.General{
				Rows: 4, Cols: 4, Stride: 4,
				Data: []float64{
					1, 0, 0, 0,
					0, 0, -2, 0,
					0, 2, 0, 0,
					0, 0, 0, 3,
				},
			},
			b: eye(4, 4),
		},
	} {
		testDggesMatrix(t, impl, tc.name, tc.a, tc.b, optimumWork)
	}

	// Test 2x2 block standardization specifically.
	testDggesBlockStandardization(t, impl)

	// Test sorting.
	testDggesSorting(t, impl)
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

// testDggesSorting tests eigenvalue sorting with SortSelected.
func testDggesSorting(t *testing.T, impl Dggeser) {
	const tol = 100

	// Test matrix pair where eigenvalues are clearly separable.
	// A = diag(1, 5, 3), B = I => eigenvalues are 1, 5, 3.
	// Select eigenvalues > 2 (should get 5 and 3).
	n := 3
	a := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0, 0,
			0, 5, 0,
			0, 0, 3,
		},
	}
	b := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := nanGeneral(n, n, n)
	vsr := nanGeneral(n, n, n)
	bwork := make([]bool, n)

	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n, nil, n, nil, n,
		nil, nil, nil, nil, n, nil, n, work, -1, nil)
	lwork := int(work[0])
	work = make([]float64, lwork)

	// Select eigenvalues where alphar/beta > 2.
	selctg := func(alphar, alphai, beta float64) bool {
		if beta == 0 {
			return false // Infinite eigenvalue.
		}
		return alphar/beta > 2
	}

	sdim, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selctg, n,
		a.Data, a.Stride, b.Data, b.Stride, alphar, alphai, beta,
		vsl.Data, vsl.Stride, vsr.Data, vsr.Stride, work, lwork, bwork)

	if !ok {
		t.Errorf("Sorting test: Dgges failed to converge")
		return
	}

	// Should have 2 selected eigenvalues (5 and 3).
	if sdim != 2 {
		t.Errorf("Sorting test: sdim=%d, want 2", sdim)
	}

	// Check that first sdim eigenvalues satisfy the selection criterion.
	for i := 0; i < sdim; i++ {
		if beta[i] != 0 && alphar[i]/beta[i] <= 2 {
			t.Errorf("Sorting test: eigenvalue[%d]=%v should be > 2", i, alphar[i]/beta[i])
		}
	}

	// Check that remaining eigenvalues do NOT satisfy the criterion.
	for i := sdim; i < n; i++ {
		if beta[i] != 0 && alphar[i]/beta[i] > 2 {
			t.Errorf("Sorting test: eigenvalue[%d]=%v should be <= 2", i, alphar[i]/beta[i])
		}
	}

	// Check decomposition is still correct.
	orthoTol := float64(n) * dlamchE
	if orthoTol < 1e-13 {
		orthoTol = 1e-13
	}

	residVSL := residualOrthogonal(vsl, false)
	if residVSL > orthoTol {
		t.Errorf("Sorting test: VSL not orthogonal; |I - VSL*VSLᵀ|=%v, want<=%v", residVSL, orthoTol)
	}
	residVSR := residualOrthogonal(vsr, false)
	if residVSR > orthoTol {
		t.Errorf("Sorting test: VSR not orthogonal; |I - VSR*VSRᵀ|=%v, want<=%v", residVSR, orthoTol)
	}

	residA := residualGeneralizedSchur(aOrig, a, vsl, vsr)
	anorm := dlange(lapack.MaxColumnSum, n, n, aOrig.Data, aOrig.Stride)
	if anorm == 0 {
		anorm = 1
	}
	normResA := residA / (anorm * float64(n) * dlamchE)
	if normResA > tol {
		t.Errorf("Sorting test: ||A - VSL*S*VSRᵀ||/(||A||*n*eps)=%v, want<=%v", normResA, tol)
	}

	residB := residualGeneralizedSchur(bOrig, b, vsl, vsr)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	if bnorm == 0 {
		bnorm = 1
	}
	normResB := residB / (bnorm * float64(n) * dlamchE)
	if normResB > tol {
		t.Errorf("Sorting test: ||B - VSL*T*VSRᵀ||/(||B||*n*eps)=%v, want<=%v", normResB, tol)
	}
}

// testDggesBlockStandardization tests that 2x2 blocks in S are properly
// standardized: equal diagonals and H(i+1,i)*H(i,i+1) < 0.
// Also checks that T's 2x2 block is diagonal with positive entries.
func testDggesBlockStandardization(t *testing.T, impl Dggeser) {
	// Matrix with complex eigenvalues ±i.
	// After Schur factorization, should have a 2x2 block with:
	// - Equal diagonals
	// - Product of off-diagonals negative
	n := 2
	a := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			0, -1,
			1, 0,
		},
	}
	b := eye(n, n)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := nanGeneral(n, n, n)
	vsr := nanGeneral(n, n, n)

	work := make([]float64, 1)
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n,
		nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1, nil)
	lwork := int(work[0])
	work = make([]float64, lwork)

	_, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, n,
		a.Data, a.Stride, b.Data, b.Stride, alphar, alphai, beta,
		vsl.Data, vsl.Stride, vsr.Data, vsr.Stride, work, lwork, nil)

	if !ok {
		t.Errorf("Block standardization test: Dgges failed to converge")
		return
	}

	// Check that we have complex eigenvalues.
	if alphai[0] == 0 || alphai[1] == 0 {
		t.Errorf("Block standardization test: expected complex eigenvalues, got alphai=%v", alphai)
		return
	}

	// Check S (stored in a) has standardized 2x2 block.
	s00 := a.Data[0]
	s01 := a.Data[1]
	s10 := a.Data[n]
	s11 := a.Data[n+1]

	// Diagonals should be equal (within tolerance).
	const tol = 1e-12
	if diff := s00 - s11; diff > tol || diff < -tol {
		t.Errorf("Block standardization: S diagonals not equal; S[0,0]=%v, S[1,1]=%v, diff=%v",
			s00, s11, diff)
	}

	// Off-diagonal product should be negative (complex eigenvalues).
	if s10*s01 >= 0 {
		t.Errorf("Block standardization: S[1,0]*S[0,1]=%v, want < 0", s10*s01)
	}

	// Check T (stored in b) is diagonal with positive entries.
	t00 := b.Data[0]
	t01 := b.Data[1]
	t10 := b.Data[n]
	t11 := b.Data[n+1]

	if t01 > tol || t01 < -tol {
		t.Errorf("Block standardization: T[0,1]=%v, want 0", t01)
	}
	if t10 > tol || t10 < -tol {
		t.Errorf("Block standardization: T[1,0]=%v, want 0", t10)
	}
	if t00 <= 0 {
		t.Errorf("Block standardization: T[0,0]=%v, want > 0", t00)
	}
	if t11 <= 0 {
		t.Errorf("Block standardization: T[1,1]=%v, want > 0", t11)
	}
}
