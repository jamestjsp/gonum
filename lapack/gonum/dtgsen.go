// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
)

// Dtgsen reorders the generalized real Schur decomposition of a real matrix
// pair (A, B) so that a selected cluster of eigenvalues appears in the leading
// diagonal blocks of the upper quasi-triangular matrix A and the upper
// triangular B.
//
// The reordering is performed using orthogonal transformations:
//
//	A := Q^T * A * Z
//	B := Q^T * B * Z
//
// The input matrices (A, B) must be in generalized real Schur canonical form
// as returned by Dhgeqz: A is block upper triangular with 1×1 and 2×2 diagonal
// blocks, and B is upper triangular.
//
// The ijob parameter specifies what is computed:
//
//	ijob=0: Reorder only, no condition estimates
//	ijob=1: Also compute reciprocal condition numbers for average of
//	        selected eigenvalues (Frobenius norm based)
//	ijob=2: Also compute reciprocal condition numbers for right and left
//	        eigenspaces (Frobenius norm based)
//	ijob=3: Also compute reciprocal condition numbers for right and left
//	        eigenspaces (1-norm based via Dlacn2)
//	ijob=4: Also compute all (1 and 2) condition numbers
//	ijob=5: ijob = 3 (kept for compatibility)
//
// If wantq is true, the left transformation Q is updated. If wantz is true,
// the right transformation Z is updated.
//
// selected specifies which eigenvalues in the cluster to reorder to the leading
// diagonal blocks. For a complex conjugate pair of eigenvalues, both must be
// selected (selected[i] and selected[i+1] must both be true).
//
// On return, m is the dimension of the specified eigenspace, and the first m
// columns of Q and Z span the corresponding left and right deflating subspaces.
//
// pl and pr are lower bounds on the reciprocal of the norm of "weights" used
// to compute the average of the selected eigenvalue group. They are valid only
// when ijob >= 1.
//
// dif is a 2-element slice where dif[0] and dif[1] are estimates of Difu and
// Difl, the separation between the matrix pairs (A11, B11) and (A22, B22).
// These are valid only when ijob >= 2.
//
// work must have length at least max(1, lwork). If lwork is -1, a workspace
// query is performed.
//
// iwork must have length at least max(1, liwork). If liwork is -1, a workspace
// query is performed.
//
// Dtgsen returns ok=false if the reordering failed because the problem is
// very ill-conditioned.
func (impl Implementation) Dtgsen(ijob int, wantq, wantz bool, selected []bool, n int,
	a []float64, lda int, b []float64, ldb int,
	alphar, alphai, beta []float64,
	q []float64, ldq int, z []float64, ldz int,
	work []float64, lwork int, iwork []int, liwork int) (m int, pl, pr float64, dif [2]float64, ok bool) {

	switch {
	case ijob < 0 || ijob > 5:
		panic("lapack: invalid ijob")
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	}

	// Quick return if possible.
	if n == 0 {
		work[0] = 1
		return 0, 0, 0, dif, true
	}

	// Compute workspace requirements.
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

	if lwork == -1 || liwork == -1 {
		if lwork == -1 {
			work[0] = float64(lwmin)
		}
		if liwork == -1 {
			iwork[0] = liwmin
		}
		return 0, 0, 0, dif, true
	}

	switch {
	case lwork < lwmin:
		panic(badLWork)
	case liwork < liwmin:
		panic("lapack: insufficient iwork length")
	case len(work) < lwork:
		panic(shortWork)
	case len(iwork) < liwork:
		panic(shortIWork)
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(selected) < n:
		panic("lapack: selected too short")
	case len(alphar) < n:
		panic("lapack: insufficient length of alphar")
	case len(alphai) < n:
		panic("lapack: insufficient length of alphai")
	case len(beta) < n:
		panic("lapack: insufficient length of beta")
	}

	bi := blas64.Implementation()
	ok = true

	// Collect selected eigenvalues at the top-left corner of (A, B).
	// Count the dimension of the specified eigenspace.
	m = 0
	pair := false
	for k := 0; k < n; k++ {
		if pair {
			pair = false
			continue
		}
		// Check if this is a 2x2 block.
		if k < n-1 && a[(k+1)*lda+k] != 0 {
			// 2x2 block - both eigenvalues must be selected together.
			if selected[k] || selected[k+1] {
				m += 2
			}
			pair = true
		} else {
			// 1x1 block.
			if selected[k] {
				m++
			}
		}
	}

	if m == 0 || m == n {
		// Nothing to do.
		goto extractEigenvalues
	}

	// Reorder blocks to collect selected eigenvalues at top-left.
	// Use a bubble-sort like approach with Dtgexc.
	{
		ks := 0      // Target position (where to move selected eigenvalues)
		pair = false // Reset pair flag

		for k := 0; k < n; k++ {
			if pair {
				pair = false
				continue
			}

			// Determine block size at position k.
			blockSize := 1
			if k < n-1 && a[(k+1)*lda+k] != 0 {
				blockSize = 2
				pair = true
			}

			// Check if this block is selected.
			isSelected := selected[k]
			if blockSize == 2 {
				isSelected = selected[k] || selected[k+1]
			}

			if isSelected {
				// Move this block to position ks.
				if k != ks {
					_, _, swapOk := impl.Dtgexc(wantq, wantz, n, a, lda, b, ldb,
						q, ldq, z, ldz, k, ks, work, lwork)
					if !swapOk {
						ok = false
						// Continue to extract eigenvalues even if reordering failed.
						goto extractEigenvalues
					}
				}
				ks += blockSize
			}
		}
	}

extractEigenvalues:
	// Extract eigenvalues from the reordered (A, B).
	// Real eigenvalues: alphar[k] = A[k,k], alphai[k] = 0, beta[k] = B[k,k]
	// Complex pairs: use Dlag2 to compute.
	{
		pair = false
		for k := 0; k < n; k++ {
			if pair {
				pair = false
				continue
			}

			if k < n-1 && a[(k+1)*lda+k] != 0 {
				// 2x2 block - complex eigenvalue pair.
				var scale1, scale2, wr1, wr2, wi float64
				scale1, scale2, wr1, wr2, wi = impl.Dlag2(
					a[k*lda+k:], lda,
					b[k*ldb+k:], ldb,
				)
				alphar[k] = wr1
				alphar[k+1] = wr2
				alphai[k] = wi
				alphai[k+1] = -wi
				beta[k] = scale1
				beta[k+1] = scale2
				pair = true
			} else {
				// 1x1 block - real eigenvalue.
				alphar[k] = a[k*lda+k]
				alphai[k] = 0
				beta[k] = b[k*ldb+k]
			}
		}
	}

	if ijob == 0 || !ok {
		work[0] = float64(lwmin)
		iwork[0] = liwmin
		return m, pl, pr, dif, ok
	}

	// Compute condition estimates.
	n1 := m
	n2 := n - m
	if n1 == 0 || n2 == 0 {
		pl = 1
		pr = 1
		dif[0] = 0
		dif[1] = 0
		work[0] = float64(lwmin)
		iwork[0] = liwmin
		return m, pl, pr, dif, ok
	}

	// Compute PL and PR (reciprocal norms for eigenvalue averaging).
	if ijob == 1 || ijob == 2 || ijob == 4 {
		// Compute Frobenius norm of the n1 x n2 blocks.
		// A12 is at A[0:n1, n1:n], B12 is at B[0:n1, n1:n].
		var sumn float64
		for i := 0; i < n1; i++ {
			for j := n1; j < n; j++ {
				sumn += a[i*lda+j] * a[i*lda+j]
				sumn += b[i*ldb+j] * b[i*ldb+j]
			}
		}
		rdscal := 0.0
		dsum := 1.0
		if sumn > 0 {
			rdscal = math.Sqrt(sumn)
			dsum = 1.0
		}

		// Solve generalized Sylvester equation to estimate PL.
		// A11*R - L*A22 = scale*A12
		// B11*R - L*B22 = scale*B12
		// Use Dtgsyl.

		// Copy A12 and B12 to work arrays.
		c := work[:n1*n2]
		f := work[n1*n2 : 2*n1*n2]
		for i := 0; i < n1; i++ {
			for j := 0; j < n2; j++ {
				c[i*n2+j] = a[i*lda+n1+j]
				f[i*n2+j] = b[i*ldb+n1+j]
			}
		}

		workTgsyl := work[2*n1*n2:]
		lworkTgsyl := lwork - 2*n1*n2
		iworkTgsyl := iwork

		_, _, tgsylOk := impl.Dtgsyl(blas.NoTrans, 0, n1, n2,
			a, lda, a[n1*lda+n1:], lda, c, n2,
			b, ldb, b[n1*ldb+n1:], ldb, f, n2,
			workTgsyl, lworkTgsyl, iworkTgsyl)
		if !tgsylOk {
			ok = false
		}

		// Compute norms for PL and PR.
		var rnorm, fnorm float64
		for i := 0; i < n1; i++ {
			for j := 0; j < n2; j++ {
				rnorm += c[i*n2+j] * c[i*n2+j]
				fnorm += f[i*n2+j] * f[i*n2+j]
			}
		}
		rnorm = math.Sqrt(rnorm)
		fnorm = math.Sqrt(fnorm)

		if rdscal != 0 {
			pl = rdscal / (rdscal*rdscal + rnorm*rnorm + fnorm*fnorm)
			pr = rdscal / (rdscal*rdscal + rnorm*rnorm + fnorm*fnorm)
		} else {
			pl = 1
			pr = 1
		}
		_ = dsum
	}

	// Compute DIF (separation estimates).
	if ijob >= 2 {
		if ijob == 2 || ijob == 4 {
			// Frobenius norm based estimate.
			// Solve the Sylvester equation to estimate Difu.
			c := work[:n1*n2]
			f := work[n1*n2 : 2*n1*n2]

			// Initialize C and F to identity-like.
			for i := range c {
				c[i] = 0
				f[i] = 0
			}
			mn := min(n1, n2)
			for i := 0; i < mn; i++ {
				c[i*n2+i] = 1
			}

			workTgsyl := work[2*n1*n2:]
			lworkTgsyl := lwork - 2*n1*n2
			iworkTgsyl := iwork

			scale, difEst, _ := impl.Dtgsyl(blas.NoTrans, 1, n1, n2,
				a, lda, a[n1*lda+n1:], lda, c, n2,
				b, ldb, b[n1*ldb+n1:], ldb, f, n2,
				workTgsyl, lworkTgsyl, iworkTgsyl)

			dif[0] = scale * difEst

			// Estimate Difl using transpose.
			for i := range c {
				c[i] = 0
				f[i] = 0
			}
			for i := 0; i < mn; i++ {
				c[i*n2+i] = 1
			}

			scale, difEst, _ = impl.Dtgsyl(blas.Trans, 1, n1, n2,
				a, lda, a[n1*lda+n1:], lda, c, n2,
				b, ldb, b[n1*ldb+n1:], ldb, f, n2,
				workTgsyl, lworkTgsyl, iworkTgsyl)

			dif[1] = scale * difEst
		} else {
			// 1-norm based estimate using Dlacn2.
			// This is more expensive but more accurate.
			nn := 2 * n1 * n2

			// Difu estimate.
			kase := 0
			isave := [3]int{}
			x := work[:nn]
			est := work[nn : nn+nn]
			for i := range x {
				x[i] = 0
			}

			for {
				est[0], kase = impl.Dlacn2(nn, work[2*nn:], x, iwork, est[0], kase, &isave)
				if kase == 0 {
					break
				}

				// Apply operator.
				c := x[:n1*n2]
				f := x[n1*n2:]
				workTgsyl := work[3*nn:]
				lworkTgsyl := lwork - 3*nn
				iworkTgsyl := iwork[n+2:]

				if kase == 1 {
					// Solve (A11, B11) * X - X * (A22, B22) = C, F.
					scale, _, _ := impl.Dtgsyl(blas.NoTrans, 0, n1, n2,
						a, lda, a[n1*lda+n1:], lda, c, n2,
						b, ldb, b[n1*ldb+n1:], ldb, f, n2,
						workTgsyl, lworkTgsyl, iworkTgsyl)
					bi.Dscal(n1*n2, scale, c, 1)
					bi.Dscal(n1*n2, scale, f, 1)
				} else {
					// Solve transpose system.
					scale, _, _ := impl.Dtgsyl(blas.Trans, 0, n1, n2,
						a, lda, a[n1*lda+n1:], lda, c, n2,
						b, ldb, b[n1*ldb+n1:], ldb, f, n2,
						workTgsyl, lworkTgsyl, iworkTgsyl)
					bi.Dscal(n1*n2, scale, c, 1)
					bi.Dscal(n1*n2, scale, f, 1)
				}
			}
			dif[0] = est[0]
			if dif[0] != 0 {
				dif[0] = 1 / dif[0]
			}

			// Difl estimate.
			kase = 0
			isave = [3]int{}
			for i := range x {
				x[i] = 0
			}

			for {
				est[0], kase = impl.Dlacn2(nn, work[2*nn:], x, iwork, est[0], kase, &isave)
				if kase == 0 {
					break
				}

				c := x[:n1*n2]
				f := x[n1*n2:]
				workTgsyl := work[3*nn:]
				lworkTgsyl := lwork - 3*nn
				iworkTgsyl := iwork[n+2:]

				if kase == 1 {
					scale, _, _ := impl.Dtgsyl(blas.Trans, 0, n1, n2,
						a, lda, a[n1*lda+n1:], lda, c, n2,
						b, ldb, b[n1*ldb+n1:], ldb, f, n2,
						workTgsyl, lworkTgsyl, iworkTgsyl)
					bi.Dscal(n1*n2, scale, c, 1)
					bi.Dscal(n1*n2, scale, f, 1)
				} else {
					scale, _, _ := impl.Dtgsyl(blas.NoTrans, 0, n1, n2,
						a, lda, a[n1*lda+n1:], lda, c, n2,
						b, ldb, b[n1*ldb+n1:], ldb, f, n2,
						workTgsyl, lworkTgsyl, iworkTgsyl)
					bi.Dscal(n1*n2, scale, c, 1)
					bi.Dscal(n1*n2, scale, f, 1)
				}
			}
			dif[1] = est[0]
			if dif[1] != 0 {
				dif[1] = 1 / dif[1]
			}
		}
	}

	work[0] = float64(lwmin)
	iwork[0] = liwmin
	return m, pl, pr, dif, ok
}
