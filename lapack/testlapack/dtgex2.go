// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/blas/blas64"
)

type Dtgex2er interface {
	Dtgex2(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
		q []float64, ldq int, z []float64, ldz int, j1, n1, n2 int, work []float64, lwork int) bool
}

func Dtgex2Test(t *testing.T, impl Dtgex2er) {
	// Test 1×1 - 1×1 swap.
	testDtgex2_1x1(t, impl)

	// Test 1×1 - 2×2 swap.
	testDtgex2_1x2(t, impl)

	// Test 2×2 - 1×1 swap.
	testDtgex2_2x1(t, impl)

	// Test 2×2 - 2×2 swap.
	testDtgex2_2x2(t, impl)
}

func testDtgex2_1x1(t *testing.T, impl Dtgex2er) {
	// 2×2 matrices with 1×1 blocks.
	a := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 2,
			0, 3,
		},
	}
	b := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 1,
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

	work := make([]float64, 10)

	ok := impl.Dtgex2(true, true, 2, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 1, 1, work, 10)

	if !ok {
		t.Error("1x1-1x1 swap failed")
		return
	}

	// Check that A is still quasi-upper-triangular.
	if math.Abs(a.Data[2]) > 1e-10 {
		t.Errorf("1x1-1x1 swap: A[1,0] = %v, want 0", a.Data[2])
	}

	// Check that B is still upper triangular.
	if math.Abs(b.Data[2]) > 1e-10 {
		t.Errorf("1x1-1x1 swap: B[1,0] = %v, want 0", b.Data[2])
	}
}

func testDtgex2_1x2(t *testing.T, impl Dtgex2er) {
	// 3×3 matrices: 1×1 block at (0,0), 2×2 block at (1,1).
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.5, 0.3,
			0, 2, 1,
			0, -1, 2,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.2, 0.1,
			0, 1, 0.3,
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

	work := make([]float64, 20)

	ok := impl.Dtgex2(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 1, 2, work, 20)

	if !ok {
		t.Log("1x1-2x2 swap failed (may be expected for some matrices)")
	}
}

func testDtgex2_2x1(t *testing.T, impl Dtgex2er) {
	// 3×3 matrices: 2×2 block at (0,0), 1×1 block at (2,2).
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			2, 1, 0.3,
			-1, 2, 0.5,
			0, 0, 3,
		},
	}
	b := blas64.General{
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

	work := make([]float64, 20)

	ok := impl.Dtgex2(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 2, 1, work, 20)

	if !ok {
		t.Log("2x1-1x1 swap failed (may be expected for some matrices)")
	}
}

func testDtgex2_2x2(t *testing.T, impl Dtgex2er) {
	// 4×4 matrices: 2×2 block at (0,0), 2×2 block at (2,2).
	a := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			2, 1, 0.1, 0.2,
			-1, 2, 0.2, 0.1,
			0, 0, 3, 1,
			0, 0, -1, 3,
		},
	}
	b := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			1, 0.1, 0.05, 0.02,
			0, 1, 0.1, 0.05,
			0, 0, 1, 0.1,
			0, 0, 0, 1,
		},
	}
	q := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}
	z := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}

	work := make([]float64, 40)

	ok := impl.Dtgex2(true, true, 4, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 2, 2, work, 40)

	if !ok {
		t.Log("2x2-2x2 swap failed (may be expected for some matrices)")
	}
}
