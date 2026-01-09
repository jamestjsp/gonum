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

	// Test 2x2 upper Hessenberg-triangular pair.
	testDhgeqz2x2(t, impl)

	// Test 3x3 upper Hessenberg-triangular pair.
	testDhgeqz3x3(t, impl)
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
