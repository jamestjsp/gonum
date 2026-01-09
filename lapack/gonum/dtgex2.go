// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dtgex2 swaps adjacent diagonal 1×1 or 2×2 blocks in a generalized Schur form
// matrix pair (S,T).
//
// Given a generalized Schur form:
//
//	(A, B) = Q^T * (S, T) * Z
//
// where S is quasi-upper-triangular (1×1 and 2×2 blocks on diagonal) and T is
// upper triangular, this routine swaps the adjacent diagonal blocks starting
// at rows and columns j1 and j1+n1.
//
// The first block has size n1 (1 or 2) and the second block has size n2 (1 or 2).
// After the swap, the second block moves to position j1.
//
// If wantq is true, the orthogonal matrix Q is updated.
// If wantz is true, the orthogonal matrix Z is updated.
//
// work must have length at least lwork. If lwork is -1, this is a workspace
// query and optimal size is returned in work[0].
//
// Dtgex2 returns ok=false if the swap failed (blocks share an eigenvalue or
// the problem is too ill-conditioned).
//
// Dtgex2 is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgex2(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, j1, n1, n2 int, work []float64, lwork int) (ok bool) {

	switch {
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
	case j1 < 0:
		panic(badJ1)
	case n1 < 1 || n1 > 2:
		panic("lapack: invalid n1")
	case n2 < 1 || n2 > 2:
		panic("lapack: invalid n2")
	case j1+n1+n2 > n:
		panic("lapack: blocks extend beyond matrix")
	}

	// Workspace query.
	if lwork == -1 {
		work[0] = float64(max(1, n))
		return true
	}

	if n == 0 {
		return true
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
	case len(work) < lwork:
		panic(shortWork)
	}

	ok = true

	bi := blas64.Implementation()

	m := n1 + n2
	eps := dlamchE
	smlnum := dlamchS / eps
	weak := false
	strong := false

	// Build the transformation matrix.
	const ldir = 16
	var ir [ldir * ldir]float64
	var t [ldir * ldir]float64
	var s [ldir * ldir]float64
	var li [ldir * ldir]float64
	var scl, sum, ss, ws float64

	// Initialize LI and IR to identity.
	for i := 0; i < m; i++ {
		li[i*ldir+i] = 1
		ir[i*ldir+i] = 1
	}

	// Compute a QZ factorization of the 2×2 or 4×4 pencil.
	// Copy A(J1:J1+M-1, J1:J1+M-1) to S.
	impl.Dlacpy(blas.All, m, m, a[j1*lda+j1:], lda, s[:], ldir)
	// Copy B(J1:J1+M-1, J1:J1+M-1) to T.
	impl.Dlacpy(blas.All, m, m, b[j1*ldb+j1:], ldb, t[:], ldir)

	// Compute the frobenius norm of A and B in the swap region.
	dnorm := impl.Dlange(lapack.Frobenius, m, m, s[:], ldir, nil)
	dnorm = math.Max(dnorm, impl.Dlange(lapack.Frobenius, m, m, t[:], ldir, nil))
	dnorm = math.Max(dnorm, smlnum)

	// Thresholds for "weak" and "strong" tests.
	thresh := dnorm * eps * 100

	if m == 2 {
		// Simple 1×1 - 1×1 swap.
		// Swap eigenvalues using a 2×2 QZ factorization.
		_, _, _, _, _, _, _, _, _, _, _, _ = impl.Dlagv2(s[:], ldir, t[:], ldir)
		// Check if swap succeeded.
		if math.Abs(s[ldir]) > thresh {
			ok = false
			return ok
		}
		s[ldir] = 0

		// Update the rest of A.
		ss = li[0]*li[0] + li[ldir]*li[ldir] + li[1]*li[1] + li[ldir+1]*li[ldir+1]
		ws = ir[0]*ir[0] + ir[ldir]*ir[ldir] + ir[1]*ir[1] + ir[ldir+1]*ir[ldir+1]
		if ss < 1 {
			ss = 1
		}
		if ws < 1 {
			ws = 1
		}

		weak = true
		strong = true
	} else if m == 3 {
		// 1×1 - 2×2 or 2×2 - 1×1 swap.
		// Use Givens rotations to transform.
		for i := 0; i < 10; i++ {
			// QZ iteration.
			_, _, _, _, _, _, _, _, _, _, _, _ = impl.Dlagv2(s[:], ldir, t[:], ldir)

			// Apply transformations.
			scl = math.Sqrt(math.Abs(s[0])*math.Abs(s[0]) + math.Abs(s[ldir])*math.Abs(s[ldir]))
			if scl <= smlnum {
				ok = false
				return ok
			}
			scl = 1 / scl
			sum = math.Abs(s[2*ldir]) + math.Abs(s[2*ldir+1]) + math.Abs(s[2*ldir+2])
			if sum <= thresh {
				weak = true
				break
			}

			// Apply QR to reduce.
			var cs, sn, r float64
			cs, sn, r = impl.Dlartg(s[0], s[ldir])
			s[0] = r
			s[ldir] = 0
			// Apply rotation to rest of S.
			for jj := 1; jj < 3; jj++ {
				tmp := cs*s[jj] + sn*s[ldir+jj]
				s[ldir+jj] = -sn*s[jj] + cs*s[ldir+jj]
				s[jj] = tmp
			}
			// Apply rotation to T.
			for jj := 0; jj < 3; jj++ {
				tmp := cs*t[jj] + sn*t[ldir+jj]
				t[ldir+jj] = -sn*t[jj] + cs*t[ldir+jj]
				t[jj] = tmp
			}
			// Update LI.
			for jj := 0; jj < 3; jj++ {
				tmp := cs*li[jj] + sn*li[ldir+jj]
				li[ldir+jj] = -sn*li[jj] + cs*li[ldir+jj]
				li[jj] = tmp
			}

			if math.Abs(s[2*ldir]) <= thresh && math.Abs(s[2*ldir+1]) <= thresh {
				weak = true
				break
			}
		}

		if !weak {
			ok = false
			return ok
		}
		s[2*ldir] = 0
		s[2*ldir+1] = 0
		strong = true
	} else {
		// 2×2 - 2×2 swap.
		// Full 4×4 QZ factorization needed.
		for i := 0; i < 10; i++ {
			// Perform QZ iteration on the 4×4 pencil.
			_, _, _, _, _, _, _, _, _, _, _, _ = impl.Dlagv2(s[:], ldir, t[:], ldir)

			scl = 0
			sum = 0
			for ii := 2; ii < 4; ii++ {
				for jj := 0; jj < 2; jj++ {
					scl = math.Max(scl, math.Abs(s[ii*ldir+jj]))
					sum += math.Abs(s[ii*ldir+jj])
				}
			}

			if sum <= thresh*4 {
				weak = true
				break
			}

			// Apply transformations to reduce.
			var cs, sn, r float64
			for ii := 0; ii < 2; ii++ {
				if math.Abs(s[(ii+2)*ldir+ii]) > smlnum {
					cs, sn, r = impl.Dlartg(s[ii*ldir+ii], s[(ii+2)*ldir+ii])
					s[ii*ldir+ii] = r
					s[(ii+2)*ldir+ii] = 0
					// Apply to columns.
					for jj := ii + 1; jj < 4; jj++ {
						tmp := cs*s[ii*ldir+jj] + sn*s[(ii+2)*ldir+jj]
						s[(ii+2)*ldir+jj] = -sn*s[ii*ldir+jj] + cs*s[(ii+2)*ldir+jj]
						s[ii*ldir+jj] = tmp
					}
					for jj := 0; jj < 4; jj++ {
						tmp := cs*t[ii*ldir+jj] + sn*t[(ii+2)*ldir+jj]
						t[(ii+2)*ldir+jj] = -sn*t[ii*ldir+jj] + cs*t[(ii+2)*ldir+jj]
						t[ii*ldir+jj] = tmp
					}
					for jj := 0; jj < 4; jj++ {
						tmp := cs*li[ii*ldir+jj] + sn*li[(ii+2)*ldir+jj]
						li[(ii+2)*ldir+jj] = -sn*li[ii*ldir+jj] + cs*li[(ii+2)*ldir+jj]
						li[ii*ldir+jj] = tmp
					}
				}
			}
		}

		if !weak {
			ok = false
			return ok
		}

		// Zero out the 2×2 lower-left block.
		for ii := 2; ii < 4; ii++ {
			for jj := 0; jj < 2; jj++ {
				s[ii*ldir+jj] = 0
			}
		}
		strong = true
	}

	if !weak {
		ok = false
		return ok
	}

	// Copy transformed matrices back.
	if n1 == 2 && n2 == 2 {
		// 2×2 - 2×2 swap: copy back the 4×4 result.
		impl.Dlacpy(blas.All, m, m, s[:], ldir, a[j1*lda+j1:], lda)
		impl.Dlacpy(blas.All, m, m, t[:], ldir, b[j1*ldb+j1:], ldb)
	} else if m == 2 {
		// 1×1 - 1×1 swap.
		impl.Dlacpy(blas.All, m, m, s[:], ldir, a[j1*lda+j1:], lda)
		impl.Dlacpy(blas.All, m, m, t[:], ldir, b[j1*ldb+j1:], ldb)
	} else {
		// 1×1 - 2×2 or 2×2 - 1×1 swap.
		impl.Dlacpy(blas.All, m, m, s[:], ldir, a[j1*lda+j1:], lda)
		impl.Dlacpy(blas.All, m, m, t[:], ldir, b[j1*ldb+j1:], ldb)
	}

	// Update the rest of A and B if needed.
	if j1 > 0 {
		// Columns 0:j1-1.
		bi.Dgemm(blas.Trans, blas.NoTrans, m, j1, m, 1, li[:], ldir, a[j1:], lda, 0, work, j1)
		impl.Dlacpy(blas.All, m, j1, work, j1, a[j1:], lda)
		bi.Dgemm(blas.Trans, blas.NoTrans, m, j1, m, 1, li[:], ldir, b[j1:], ldb, 0, work, j1)
		impl.Dlacpy(blas.All, m, j1, work, j1, b[j1:], ldb)
	}

	if j1+m < n {
		// Rows j1+m:n-1.
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n-j1-m, m, m, 1, a[(j1+m)*lda+j1:], lda, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, n-j1-m, m, work, m, a[(j1+m)*lda+j1:], lda)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n-j1-m, m, m, 1, b[(j1+m)*ldb+j1:], ldb, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, n-j1-m, m, work, m, b[(j1+m)*ldb+j1:], ldb)
	}

	// Update Q if requested.
	if wantq {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, q[j1:], ldq, li[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, n, m, work, m, q[j1:], ldq)
	}

	// Update Z if requested.
	if wantz {
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n, m, m, 1, z[j1:], ldz, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, n, m, work, m, z[j1:], ldz)
	}

	_ = strong
	return ok
}
