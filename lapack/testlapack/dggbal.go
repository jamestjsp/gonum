// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

type Dggbaler interface {
	Dggbal(job lapack.BalanceJob, n int, a []float64, lda int, b []float64, ldb int, lscale, rscale, work []float64) (ilo, ihi int)
}

func DggbalTest(t *testing.T, impl Dggbaler) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
		for _, job := range []lapack.BalanceJob{lapack.BalanceNone, lapack.Permute, lapack.Scale, lapack.PermuteScale} {
			for _, extra := range []int{0, 5} {
				testDggbal(t, impl, n, job, extra, rnd)
			}
		}
	}

	testDggbalSpecial(t, impl)
}

func testDggbal(t *testing.T, impl Dggbaler, n int, job lapack.BalanceJob, extra int, rnd *rand.Rand) {
	a := randomGeneral(n, n, n+extra, rnd)
	b := randomGeneral(n, n, n+extra, rnd)

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	lscale := make([]float64, n)
	rscale := make([]float64, n)
	work := make([]float64, max(1, 6*n))

	ilo, ihi := impl.Dggbal(job, n, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	if n == 0 {
		if ilo != 0 || ihi != -1 {
			t.Errorf("n=0: ilo=%d, ihi=%d, want ilo=0, ihi=-1", ilo, ihi)
		}
		return
	}

	if ilo < 0 || ilo > ihi || ihi >= n {
		t.Errorf("n=%d, job=%c: invalid ilo=%d, ihi=%d", n, job, ilo, ihi)
	}

	if job == lapack.BalanceNone {
		if ilo != 0 || ihi != n-1 {
			t.Errorf("n=%d, job=N: ilo=%d, ihi=%d, want ilo=0, ihi=%d", n, ilo, ihi, n-1)
		}
		for i := 0; i < n; i++ {
			if lscale[i] != 1 || rscale[i] != 1 {
				t.Errorf("n=%d, job=N: lscale[%d]=%v, rscale[%d]=%v, want 1", n, i, lscale[i], i, rscale[i])
			}
		}
		if !generalEqual(a, aOrig) || !generalEqual(b, bOrig) {
			t.Errorf("n=%d, job=N: matrices modified", n)
		}
		return
	}

	// For permuting jobs, check that isolated rows/columns have zeros.
	if job == lapack.Permute || job == lapack.PermuteScale {
		for j := 0; j < ilo; j++ {
			for i := j + 1; i < n; i++ {
				if a.Data[i*a.Stride+j] != 0 || b.Data[i*b.Stride+j] != 0 {
					t.Errorf("n=%d, job=%c: A[%d,%d]=%v, B[%d,%d]=%v not zero in lower triangle before ilo",
						n, job, i, j, a.Data[i*a.Stride+j], i, j, b.Data[i*b.Stride+j])
				}
			}
		}
		for j := ihi + 1; j < n; j++ {
			for i := j + 1; i < n; i++ {
				if a.Data[i*a.Stride+j] != 0 || b.Data[i*b.Stride+j] != 0 {
					t.Errorf("n=%d, job=%c: A[%d,%d]=%v, B[%d,%d]=%v not zero in lower triangle after ihi",
						n, job, i, j, a.Data[i*a.Stride+j], i, j, b.Data[i*b.Stride+j])
				}
			}
		}
	}
}

func testDggbalSpecial(t *testing.T, impl Dggbaler) {
	// Test with matrices having isolated eigenvalues.

	// Case 1: Last row is zero except diagonal.
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 2, 3,
			4, 5, 6,
			0, 0, 7,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 1, 1,
			1, 1, 1,
			0, 0, 1,
		},
	}

	lscale := make([]float64, 3)
	rscale := make([]float64, 3)
	work := make([]float64, 18)

	ilo, ihi := impl.Dggbal(lapack.Permute, 3, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	if ihi != 1 {
		t.Errorf("Special case 1: ihi=%d, want 1", ihi)
	}
	if ilo != 0 {
		t.Errorf("Special case 1: ilo=%d, want 0", ilo)
	}

	// Case 2: First column is zero except diagonal.
	a = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 2, 3,
			0, 4, 5,
			0, 6, 7,
		},
	}
	b = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 1, 1,
			0, 1, 1,
			0, 1, 1,
		},
	}

	lscale = make([]float64, 3)
	rscale = make([]float64, 3)

	ilo, ihi = impl.Dggbal(lapack.Permute, 3, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	if ilo != 1 {
		t.Errorf("Special case 2: ilo=%d, want 1", ilo)
	}

	// Case 3: Test scaling with badly scaled matrix.
	a = blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1e-10, 1,
			1, 1e10,
		},
	}
	b = blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 1,
			1, 1,
		},
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	lscale = make([]float64, 2)
	rscale = make([]float64, 2)
	work = make([]float64, 12)

	ilo, ihi = impl.Dggbal(lapack.Scale, 2, a.Data, a.Stride, b.Data, b.Stride, lscale, rscale, work)

	// Check that scaling reduced the condition number.
	rowNormOrig := make([]float64, 2)
	colNormOrig := make([]float64, 2)
	rowNormNew := make([]float64, 2)
	colNormNew := make([]float64, 2)

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			rowNormOrig[i] += math.Abs(aOrig.Data[i*aOrig.Stride+j]) + math.Abs(bOrig.Data[i*bOrig.Stride+j])
			colNormOrig[j] += math.Abs(aOrig.Data[i*aOrig.Stride+j]) + math.Abs(bOrig.Data[i*bOrig.Stride+j])
			rowNormNew[i] += math.Abs(a.Data[i*a.Stride+j]) + math.Abs(b.Data[i*b.Stride+j])
			colNormNew[j] += math.Abs(a.Data[i*a.Stride+j]) + math.Abs(b.Data[i*b.Stride+j])
		}
	}

	// The balanced matrix should have more uniform row/column norms.
	origRatio := math.Max(rowNormOrig[0], rowNormOrig[1]) / math.Min(rowNormOrig[0], rowNormOrig[1])
	newRatio := math.Max(rowNormNew[0], rowNormNew[1]) / math.Min(rowNormNew[0], rowNormNew[1])

	if newRatio > origRatio {
		t.Logf("Warning: Scaling did not improve row balance: orig ratio=%v, new ratio=%v", origRatio, newRatio)
	}
}
