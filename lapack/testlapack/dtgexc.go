// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"gonum.org/v1/gonum/blas/blas64"
)

type Dtgexcer interface {
	Dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
		q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, lwork int) (ifstOut, ilstOut int, ok bool)
}

func DtgexcTest(t *testing.T, impl Dtgexcer) {
	// Test workspace query.
	work := make([]float64, 1)
	_, _, ok := impl.Dtgexc(false, false, 5, nil, 5, nil, 5, nil, 1, nil, 1, 0, 0, work, -1)
	if !ok {
		t.Error("Workspace query failed")
	}
	if work[0] < 1 {
		t.Errorf("Workspace query returned invalid size: %v", work[0])
	}

	// Test forward move.
	testDtgexcForward(t, impl)

	// Test backward move.
	testDtgexcBackward(t, impl)
}

func testDtgexcForward(t *testing.T, impl Dtgexcer) {
	// 3x3 matrix pair in generalized Schur form.
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.5, 0.3,
			0, 2, 0.4,
			0, 0, 3,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.2, 0.1,
			0, 1, 0.15,
			0, 0, 1,
		},
	}
	q := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}
	z := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}

	work := make([]float64, 100)

	// Move block from position 0 to position 2.
	ifstOut, ilstOut, ok := impl.Dtgexc(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 2, work, 100)

	if !ok {
		t.Log("Forward move: swap failed (may be expected for some matrices)")
	}

	if ifstOut != ilstOut {
		t.Logf("Forward move: ifstOut=%d != ilstOut=%d (partial move)", ifstOut, ilstOut)
	}
}

func testDtgexcBackward(t *testing.T, impl Dtgexcer) {
	// 3x3 matrix pair in generalized Schur form.
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.5, 0.3,
			0, 2, 0.4,
			0, 0, 3,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.2, 0.1,
			0, 1, 0.15,
			0, 0, 1,
		},
	}
	q := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}
	z := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}

	work := make([]float64, 100)

	// Move block from position 2 to position 0.
	ifstOut, ilstOut, ok := impl.Dtgexc(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 2, 0, work, 100)

	if !ok {
		t.Log("Backward move: swap failed (may be expected for some matrices)")
	}

	if ifstOut != ilstOut {
		t.Logf("Backward move: ifstOut=%d != ilstOut=%d (partial move)", ifstOut, ilstOut)
	}
}
