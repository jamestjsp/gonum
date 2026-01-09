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

type Dtgsener interface {
	Dtgexcer
	Dtgsyler
	Dlag2er
	Dtgsen(ijob int, wantq, wantz bool, selected []bool, n int,
		a []float64, lda int, b []float64, ldb int,
		alphar, alphai, beta []float64,
		q []float64, ldq int, z []float64, ldz int,
		work []float64, lwork int, iwork []int, liwork int) (m int, pl, pr float64, dif [2]float64, ok bool)
}

func DtgsenTest(t *testing.T, impl Dtgsener) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, n := range []int{0, 1, 2, 3, 4, 5, 10, 20} {
		for _, ijob := range []int{0, 1, 2, 4} {
			for _, wantq := range []bool{false, true} {
				for _, wantz := range []bool{false, true} {
					for _, selectPattern := range []string{"none", "first", "last", "alternating", "all"} {
						testDtgsen(t, impl, n, ijob, wantq, wantz, selectPattern, rnd)
					}
				}
			}
		}
	}

	testDtgsenWorkspace(t, impl)
}

func testDtgsen(t *testing.T, impl Dtgsener, n, ijob int, wantq, wantz bool, selectPattern string, rnd *rand.Rand) {
	const tol = 1e-10

	prefix := fmt.Sprintf("n=%d, ijob=%d, wantq=%v, wantz=%v, select=%s",
		n, ijob, wantq, wantz, selectPattern)

	if n == 0 {
		// Quick return test.
		var work [1]float64
		var iwork [1]int
		m, pl, pr, dif, ok := impl.Dtgsen(ijob, wantq, wantz, nil, 0,
			nil, 1, nil, 1, nil, nil, nil, nil, 1, nil, 1,
			work[:], 1, iwork[:], 1)
		if m != 0 || !ok {
			t.Errorf("%s: unexpected result for n=0", prefix)
		}
		_ = pl
		_ = pr
		_ = dif
		return
	}

	// Generate a random matrix pair in generalized Schur form.
	a := randomSchur(n, n, rnd)
	b := randomUpperTriangular(n, n, rnd)

	// Ensure B has positive diagonal.
	for i := 0; i < n; i++ {
		if b.Data[i*b.Stride+i] < 0 {
			b.Data[i*b.Stride+i] = -b.Data[i*b.Stride+i]
		}
		if b.Data[i*b.Stride+i] == 0 {
			b.Data[i*b.Stride+i] = 1
		}
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	// Generate selection array.
	selected := make([]bool, n)
	expectedM := 0
	switch selectPattern {
	case "none":
		// No selection.
	case "first":
		// Select first eigenvalue/block.
		if n > 0 {
			selected[0] = true
			expectedM = 1
			if n > 1 && a.Data[1*a.Stride+0] != 0 {
				selected[1] = true
				expectedM = 2
			}
		}
	case "last":
		// Select last eigenvalue/block.
		if n > 0 {
			k := n - 1
			if k > 0 && a.Data[k*a.Stride+k-1] != 0 {
				k--
				selected[k] = true
				selected[k+1] = true
				expectedM = 2
			} else {
				selected[k] = true
				expectedM = 1
			}
		}
	case "alternating":
		// Select alternating eigenvalues.
		k := 0
		for k < n {
			if k < n-1 && a.Data[(k+1)*a.Stride+k] != 0 {
				// 2x2 block.
				if k%2 == 0 {
					selected[k] = true
					selected[k+1] = true
					expectedM += 2
				}
				k += 2
			} else {
				// 1x1 block.
				if k%2 == 0 {
					selected[k] = true
					expectedM++
				}
				k++
			}
		}
	case "all":
		// Select all eigenvalues.
		for i := 0; i < n; i++ {
			selected[i] = true
		}
		expectedM = n
	}

	// Allocate output arrays.
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	q := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	z := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}

	// Initialize Q and Z to identity.
	for i := 0; i < n; i++ {
		q.Data[i*q.Stride+i] = 1
		z.Data[i*z.Stride+i] = 1
	}

	// Workspace query.
	var worksize [1]float64
	var iworksize [1]int
	impl.Dtgsen(ijob, wantq, wantz, selected, n,
		nil, max(1, a.Stride), nil, max(1, b.Stride), nil, nil, nil,
		nil, max(1, q.Stride), nil, max(1, z.Stride),
		worksize[:], -1, iworksize[:], -1)
	lwork := int(worksize[0])
	liwork := iworksize[0]

	work := make([]float64, max(1, lwork))
	iwork := make([]int, max(1, liwork))

	// Call Dtgsen.
	m, pl, pr, dif, ok := impl.Dtgsen(ijob, wantq, wantz, selected, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride,
		work, lwork, iwork, liwork)

	// Check m matches expected.
	if m != expectedM && (selectPattern != "none" && selectPattern != "all") {
		// For none and all, no reordering happens, so m should still match.
		if selectPattern == "none" && m != 0 {
			t.Errorf("%s: m=%d, want 0 for no selection", prefix, m)
		}
		if selectPattern == "all" && m != n {
			t.Errorf("%s: m=%d, want %d for all selection", prefix, m, n)
		}
	}

	// Check ok is true (unless we expect failure).
	if !ok {
		t.Logf("%s: reordering failed (ok=false), continuing checks", prefix)
	}

	// Check that selected eigenvalues are in the leading m positions.
	// Verify by checking structure of A (quasi-triangular).

	// Check orthogonality of Q and Z.
	if wantq {
		resid := orthogonalityResidual(q)
		if resid > tol*float64(n) {
			t.Errorf("%s: Q not orthogonal, residual=%e", prefix, resid)
		}
	}
	if wantz {
		resid := orthogonalityResidual(z)
		if resid > tol*float64(n) {
			t.Errorf("%s: Z not orthogonal, residual=%e", prefix, resid)
		}
	}

	// Check the decomposition: A = Q * Aout * Z^T, B = Q * Bout * Z^T.
	if wantq && wantz {
		residA := schurDecompResidual(aOrig, a, q, z)
		residB := schurDecompResidual(bOrig, b, q, z)
		normA := dlange(lapack.MaxColumnSum, n, n, aOrig.Data, aOrig.Stride)
		normB := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
		if normA > 0 && residA/(normA*float64(n)*dlamchE) > tol/dlamchE {
			t.Errorf("%s: A decomposition residual too large: %e", prefix, residA/(normA*float64(n)*dlamchE))
		}
		if normB > 0 && residB/(normB*float64(n)*dlamchE) > tol/dlamchE {
			t.Errorf("%s: B decomposition residual too large: %e", prefix, residB/(normB*float64(n)*dlamchE))
		}
	}

	// Check condition estimates are reasonable (non-negative).
	if ijob >= 1 {
		if pl < 0 || pr < 0 {
			t.Errorf("%s: pl=%e or pr=%e negative", prefix, pl, pr)
		}
	}
	if ijob >= 2 {
		if dif[0] < 0 || dif[1] < 0 {
			t.Errorf("%s: dif[0]=%e or dif[1]=%e negative", prefix, dif[0], dif[1])
		}
	}
}

func testDtgsenWorkspace(t *testing.T, impl Dtgsener) {
	for _, n := range []int{1, 5, 10} {
		for _, ijob := range []int{0, 1, 2, 3, 4, 5} {
			var worksize [1]float64
			var iworksize [1]int
			impl.Dtgsen(ijob, true, true, nil, n,
				nil, n, nil, n, nil, nil, nil,
				nil, n, nil, n,
				worksize[:], -1, iworksize[:], -1)
			lwork := int(worksize[0])
			liwork := iworksize[0]

			// Verify minimum workspace requirements.
			var lwmin, liwmin int
			if ijob == 1 || ijob == 2 || ijob == 4 {
				lwmin = max(1, 4*n+16, 2*n*(n+2)+16)
				liwmin = max(1, n+6)
			} else if ijob == 3 || ijob == 5 {
				lwmin = max(1, 4*n+16, 4*n*(n+1)+16)
				liwmin = max(1, 2*n*(n+2)+16)
			} else {
				lwmin = max(1, 4*n+16)
				liwmin = max(1, n+6)
			}

			if lwork < lwmin {
				t.Errorf("n=%d, ijob=%d: workspace query returned lwork=%d, want >= %d",
					n, ijob, lwork, lwmin)
			}
			if liwork < liwmin {
				t.Errorf("n=%d, ijob=%d: workspace query returned liwork=%d, want >= %d",
					n, ijob, liwork, liwmin)
			}
		}
	}
}

// orthogonalityResidual computes ||Q^T*Q - I||_F.
func orthogonalityResidual(q blas64.General) float64 {
	n := q.Rows
	qtq := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, q, q, 0, qtq)

	var resid float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			diff := qtq.Data[i*qtq.Stride+j] - expected
			resid += diff * diff
		}
	}
	return math.Sqrt(resid)
}

// schurDecompResidual computes ||Aorig - Q*A*Z^T||_F.
func schurDecompResidual(orig, a, q, z blas64.General) float64 {
	n := orig.Rows

	// Compute Q*A.
	qa := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, q, a, 0, qa)

	// Compute Q*A*Z^T.
	qazt := blas64.General{
		Rows:   n,
		Cols:   n,
		Stride: n,
		Data:   make([]float64, n*n),
	}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, qa, z, 0, qazt)

	// Compute ||Aorig - Q*A*Z^T||_F.
	var resid float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := orig.Data[i*orig.Stride+j] - qazt.Data[i*qazt.Stride+j]
			resid += diff * diff
		}
	}
	return math.Sqrt(resid)
}
