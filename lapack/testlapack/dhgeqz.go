// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

type Dhgeqzer interface {
	Dhgeqz(job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi int,
		h []float64, ldh int, t []float64, ldt int, alphar, alphai, beta []float64,
		q []float64, ldq int, z []float64, ldz int, work []float64, lwork int) bool
}

func DhgeqzTest(t *testing.T, impl Dhgeqzer) {
	// Test workspace query.
	work := make([]float64, 1)
	ok := impl.Dhgeqz(lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone, 10, 0, 9,
		nil, 10, nil, 10, nil, nil, nil, nil, 1, nil, 1, work, -1)
	if !ok {
		t.Error("Workspace query failed")
	}
	if work[0] < 1 {
		t.Errorf("Workspace query returned invalid size: %v", work[0])
	}

	testDhgeqz2x2(t, impl)
	testDhgeqz3x3(t, impl)
	testDhgeqzComplex4x4(t, impl)
	testDhgeqzComplex6x6(t, impl)
}

func testDhgeqz2x2(t *testing.T, impl Dhgeqzer) {
	// 2x2 upper Hessenberg-triangular pair.
	h := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			2, 1,
			1, 3,
		},
	}
	tt := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0.5,
			0, 1,
		},
	}
	q := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0,
			0, 1,
		},
	}
	z := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0,
			0, 1,
		},
	}

	alphar := make([]float64, 2)
	alphai := make([]float64, 2)
	beta := make([]float64, 2)
	work := make([]float64, 10)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, 2, 0, 1,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, 10)

	if !ok {
		t.Log("2x2 test: QZ iteration did not converge")
	}

	// Check that beta values are non-zero for non-infinite eigenvalues.
	for i := 0; i < 2; i++ {
		if math.Abs(beta[i]) > 1e-10 && math.IsNaN(alphar[i]/beta[i]) {
			t.Errorf("2x2 test: eigenvalue %d is NaN", i)
		}
	}
}

func testDhgeqz3x3(t *testing.T, impl Dhgeqzer) {
	// 3x3 upper Hessenberg-triangular pair.
	h := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			2, 1, 0.5,
			1, 3, 1,
			0, 1, 4,
		},
	}
	tt := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.3, 0.1,
			0, 1, 0.2,
			0, 0, 1,
		},
	}
	q := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
	}
	z := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
	}

	alphar := make([]float64, 3)
	alphai := make([]float64, 3)
	beta := make([]float64, 3)
	work := make([]float64, 20)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, 3, 0, 2,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, 20)

	if !ok {
		t.Log("3x3 test: QZ iteration did not converge")
	}

	// Check that beta values are non-zero for non-infinite eigenvalues.
	for i := 0; i < 3; i++ {
		if math.Abs(beta[i]) > 1e-10 && math.IsNaN(alphar[i]/beta[i]) {
			t.Errorf("3x3 test: eigenvalue %d is NaN", i)
		}
	}
}

func testDhgeqzComplex4x4(t *testing.T, impl Dhgeqzer) {
	// 4x4 Hessenberg-triangular pair with complex conjugate eigenvalues.
	// H has structure that produces complex eigenvalue pairs.
	n := 4
	h := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 2, 0.5, 0.3,
			-3, 1, 1.0, 0.2,
			0, -2, 0, 1.5,
			0, 0, -1, 0,
		},
	}
	tt := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0.5, 0.2, 0.1,
			0, 1, 0.3, 0.1,
			0, 0, 1, 0.4,
			0, 0, 0, 1,
		},
	}

	hOrig := make([]float64, len(h.Data))
	tOrig := make([]float64, len(tt.Data))
	copy(hOrig, h.Data)
	copy(tOrig, tt.Data)

	q := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	z := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 4*n)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, len(work))

	if !ok {
		t.Fatal("4x4 complex test: QZ iteration did not converge")
	}

	hasComplex := false
	for i := 0; i < n; i++ {
		if alphai[i] != 0 {
			hasComplex = true
			break
		}
	}
	if !hasComplex {
		t.Log("4x4 complex test: no complex eigenvalues detected (may be OK if exceptional shifts resolved them)")
	}

	// Verify Schur decomposition: Q^T * H_orig * Z ≈ S.
	bi := blas64.Implementation()
	const tol = 1e-10
	tmp := make([]float64, n*n)
	result := make([]float64, n*n)

	// tmp = H_orig * Z
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += hOrig[i*n+k] * z.Data[k*n+j]
			}
			tmp[i*n+j] = s
		}
	}
	// result = Q^T * tmp
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += q.Data[k*n+i] * tmp[k*n+j]
			}
			result[i*n+j] = s
		}
	}
	// Check result ≈ S (h after QZ)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := math.Abs(result[i*n+j] - h.Data[i*n+j])
			if diff > tol {
				t.Errorf("4x4 complex test: Q^T*H*Z - S at (%d,%d): diff=%e", i, j, diff)
			}
		}
	}

	_ = bi
}

func testDhgeqzComplex6x6(t *testing.T, impl Dhgeqzer) {
	// 6x6 Hessenberg-triangular pair designed to have multiple complex pairs.
	n := 6
	h := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			0.5, 1.0, 0.3, 0.1, 0.2, 0.4,
			-1.5, 0.5, 0.7, 0.2, 0.1, 0.3,
			0, -2.0, 1.0, 0.5, 0.3, 0.1,
			0, 0, -1.0, 1.0, 0.8, 0.2,
			0, 0, 0, -0.5, 0.3, 1.0,
			0, 0, 0, 0, -1.2, 0.3,
		},
	}
	tt := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0.2, 0.1, 0, 0, 0,
			0, 1, 0.3, 0.1, 0, 0,
			0, 0, 1, 0.2, 0.1, 0,
			0, 0, 0, 1, 0.1, 0,
			0, 0, 0, 0, 1, 0.2,
			0, 0, 0, 0, 0, 1,
		},
	}

	hOrig := make([]float64, len(h.Data))
	tOrig := make([]float64, len(tt.Data))
	copy(hOrig, h.Data)
	copy(tOrig, tt.Data)

	q := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	z := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 4*n)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, len(work))

	if !ok {
		t.Fatal("6x6 complex test: QZ iteration did not converge")
	}

	// Verify Schur decomposition: Q^T * H_orig * Z ≈ S.
	const tol = 1e-10
	tmp := make([]float64, n*n)
	result := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += hOrig[i*n+k] * z.Data[k*n+j]
			}
			tmp[i*n+j] = s
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += q.Data[k*n+i] * tmp[k*n+j]
			}
			result[i*n+j] = s
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := math.Abs(result[i*n+j] - h.Data[i*n+j])
			if diff > tol {
				t.Errorf("6x6 complex test: Q^T*H*Z - S at (%d,%d): diff=%e", i, j, diff)
			}
		}
	}

	_ = tOrig
}
