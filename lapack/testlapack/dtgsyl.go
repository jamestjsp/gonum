// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

type Dtgsyler interface {
	Dtgsy2er
	Dtgsyl(trans blas.Transpose, ijob, m, n int,
		a []float64, lda int, b []float64, ldb int, c []float64, ldc int,
		d []float64, ldd int, e []float64, lde int, f []float64, ldf int,
		work []float64, lwork int, iwork []int) (scale, dif float64, ok bool)
}

func DtgsylTest(t *testing.T, impl Dtgsyler) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, m := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
		for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, ijob := range []int{0, 1, 2} {
					for _, extra := range []int{0, 3} {
						testDtgsyl(t, impl, m, n, trans, ijob, extra, rnd)
					}
				}
			}
		}
	}

	testDtgsylWorkspace(t, impl)
}

func testDtgsyl(t *testing.T, impl Dtgsyler, m, n int, trans blas.Transpose, ijob, extra int, rnd *rand.Rand) {
	const tol = 1e-10

	prefix := fmt.Sprintf("m=%v, n=%v, trans=%c, ijob=%d, extra=%v",
		m, n, trans, ijob, extra)

	// Generate random quasi-triangular matrices A (m×m) and upper triangular B (n×n).
	a := randomSchur(m, m+extra, rnd)
	b := randomUpperTriangular(n, n+extra, rnd)

	// Generate random upper triangular matrices D (m×m) and E (n×n).
	d := randomUpperTriangular(m, m+extra, rnd)
	e := randomUpperTriangular(n, n+extra, rnd)

	// Generate random RHS matrices C, F (m×n).
	c := randomGeneral(m, n, n+extra, rnd)
	f := randomGeneral(m, n, n+extra, rnd)
	cOrig := cloneGeneral(c)
	fOrig := cloneGeneral(f)

	// Keep originals for residual check.
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)
	dOrig := cloneGeneral(d)
	eOrig := cloneGeneral(e)

	// Workspace query.
	var worksize [1]float64
	impl.Dtgsyl(trans, ijob, m, n,
		nil, max(1, a.Stride), nil, max(1, b.Stride), nil, max(1, c.Stride),
		nil, max(1, d.Stride), nil, max(1, e.Stride), nil, max(1, f.Stride),
		worksize[:], -1, nil)
	lwork := int(worksize[0])
	work := make([]float64, max(1, lwork))
	iwork := make([]int, m+n+6)

	// Call Dtgsyl.
	scale, dif, ok := impl.Dtgsyl(trans, ijob, m, n,
		a.Data, a.Stride, b.Data, b.Stride, c.Data, c.Stride,
		d.Data, d.Stride, e.Data, e.Stride, f.Data, f.Stride,
		work, lwork, iwork)

	// Check that A, B, D, E are not modified.
	if !generalEqual(a, aOrig) {
		t.Errorf("%v: A was modified", prefix)
	}
	if !generalEqual(b, bOrig) {
		t.Errorf("%v: B was modified", prefix)
	}
	if !generalEqual(d, dOrig) {
		t.Errorf("%v: D was modified", prefix)
	}
	if !generalEqual(e, eOrig) {
		t.Errorf("%v: E was modified", prefix)
	}

	// Quick return for empty matrices.
	if m == 0 || n == 0 {
		if scale != 1 {
			t.Errorf("%v: scale=%v, want 1 for empty matrix", prefix, scale)
		}
		return
	}

	// Verify the solution by computing the residual.
	resid := dtgsy2Residual(aOrig, bOrig, cOrig, dOrig, eOrig, fOrig, c, f, trans, scale)

	anorm := dlange(lapack.MaxColumnSum, m, m, aOrig.Data, aOrig.Stride)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	dnorm := dlange(lapack.MaxColumnSum, m, m, dOrig.Data, dOrig.Stride)
	enorm := dlange(lapack.MaxColumnSum, n, n, eOrig.Data, eOrig.Stride)
	rnorm := dlange(lapack.MaxColumnSum, m, n, c.Data, c.Stride)
	lnorm := dlange(lapack.MaxColumnSum, m, n, f.Data, f.Stride)
	cnorm := dlange(lapack.MaxColumnSum, m, n, cOrig.Data, cOrig.Stride)
	fnorm := dlange(lapack.MaxColumnSum, m, n, fOrig.Data, fOrig.Stride)

	denom := (anorm+dnorm)*rnorm + (bnorm+enorm)*lnorm + cnorm + fnorm
	if denom == 0 {
		denom = 1
	}
	normRes := resid / (denom * float64(max(m, n)) * dlamchE)

	if normRes > tol/dlamchE {
		t.Errorf("%v: residual too large: %v, want <= %v (ok=%v, scale=%v)", prefix, normRes, tol/dlamchE, ok, scale)
	}

	// Check Dif estimate is non-negative.
	if ijob >= 1 && trans == blas.NoTrans && dif < 0 {
		t.Errorf("%v: dif=%v, want >= 0", prefix, dif)
	}
}

func testDtgsylWorkspace(t *testing.T, impl Dtgsyler) {
	// Test workspace query.
	for _, m := range []int{1, 5, 10} {
		for _, n := range []int{1, 5, 10} {
			for _, ijob := range []int{0, 1, 2} {
				var worksize [1]float64
				impl.Dtgsyl(blas.NoTrans, ijob, m, n,
					nil, m, nil, n, nil, n,
					nil, m, nil, n, nil, n,
					worksize[:], -1, nil)
				lwork := int(worksize[0])

				var expectedMin int
				if ijob == 1 || ijob == 2 {
					expectedMin = 2 * m * n
				} else {
					expectedMin = 1
				}

				if lwork < expectedMin {
					t.Errorf("m=%d, n=%d, ijob=%d: workspace query returned %d, want >= %d",
						m, n, ijob, lwork, expectedMin)
				}
			}
		}
	}
}

func dtgsylResidual(A, B, C, D, E, F, R, L blas64.General, trans blas.Transpose, scale float64) float64 {
	m := A.Rows
	n := B.Rows

	if m == 0 || n == 0 {
		return 0
	}

	var resid1, resid2 float64

	if trans == blas.NoTrans {
		// Equation 1: A*R - L*B = scale*C
		res1 := zeros(m, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, A, R, 0, res1)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, L, B, 1, res1)
		for i := range m {
			for j := range n {
				res1.Data[i*res1.Stride+j] -= scale * C.Data[i*C.Stride+j]
			}
		}
		resid1 = dlange(lapack.MaxColumnSum, m, n, res1.Data, res1.Stride)

		// Equation 2: D*R - L*E = scale*F
		res2 := zeros(m, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, D, R, 0, res2)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, L, E, 1, res2)
		for i := range m {
			for j := range n {
				res2.Data[i*res2.Stride+j] -= scale * F.Data[i*F.Stride+j]
			}
		}
		resid2 = dlange(lapack.MaxColumnSum, m, n, res2.Data, res2.Stride)
	} else {
		// Equation 1: A^T*R + D^T*L = scale*C
		res1 := zeros(m, n, n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, A, R, 0, res1)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, D, L, 1, res1)
		for i := range m {
			for j := range n {
				res1.Data[i*res1.Stride+j] -= scale * C.Data[i*C.Stride+j]
			}
		}
		resid1 = dlange(lapack.MaxColumnSum, m, n, res1.Data, res1.Stride)

		// Equation 2: R*B^T + L*E^T = -scale*F
		res2 := zeros(m, n, n)
		blas64.Gemm(blas.NoTrans, blas.Trans, 1, R, B, 0, res2)
		blas64.Gemm(blas.NoTrans, blas.Trans, 1, L, E, 1, res2)
		for i := range m {
			for j := range n {
				res2.Data[i*res2.Stride+j] += scale * F.Data[i*F.Stride+j]
			}
		}
		resid2 = dlange(lapack.MaxColumnSum, m, n, res2.Data, res2.Stride)
	}

	return math.Max(resid1, resid2)
}
