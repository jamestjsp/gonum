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
	var sum float64

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
		//
		// For upper triangular (A, B) = ([[a11, a12], [0, a22]], [[b11, b12], [0, b22]]),
		// we want to swap eigenvalues λ1 = a11/b11 and λ2 = a22/b22.
		//
		// The eigenvector for λ2 satisfies (A - λ2*B)*v = 0:
		//   (a11*b22 - a22*b11)*v1 + (a12*b22 - a22*b12)*v2 = 0
		//
		// So v ∝ [a12*b22 - a22*b12, a22*b11 - a11*b22]^T
		//
		// We construct Z so its first column is this eigenvector (normalized).
		// Then Q^T*A*Z will have λ2 at position (0,0).

		a11 := s[0]
		a12 := s[1]
		a22 := s[ldir+1]
		b11 := t[0]
		b12 := t[1]
		b22 := t[ldir+1]

		// Check if swap is possible (eigenvalues must be distinct).
		denom := a11*b22 - a22*b11
		if math.Abs(denom) <= eps*math.Max(math.Abs(a11)*math.Abs(b22), math.Abs(a22)*math.Abs(b11)) {
			ok = false
			return ok
		}

		// Eigenvector for λ2 is proportional to [f, g]^T where:
		//   f = a12*b22 - a22*b12
		//   g = a22*b11 - a11*b22 = -denom
		f := a12*b22 - a22*b12
		g := -denom

		csz, snz, _ := impl.Dlartg(f, g)

		// Apply Z from right: Z mixes columns 0 and 1.
		// new col 0 = csz*col0 + snz*col1
		// new col 1 = -snz*col0 + csz*col1
		s[0] = csz*a11 + snz*a12
		s[1] = -snz*a11 + csz*a12
		s[ldir] = snz * a22
		s[ldir+1] = csz * a22

		t[0] = csz*b11 + snz*b12
		t[1] = -snz*b11 + csz*b12
		t[ldir] = snz * b22
		t[ldir+1] = csz * b22

		// Apply Q from left to zero out s[ldir].
		csq, snq, _ := impl.Dlartg(s[0], s[ldir])

		sa := s[0]
		s[0] = csq*sa + snq*s[ldir]
		s[ldir] = -snq*sa + csq*s[ldir]
		sa = s[1]
		s[1] = csq*sa + snq*s[ldir+1]
		s[ldir+1] = -snq*sa + csq*s[ldir+1]

		sb := t[0]
		t[0] = csq*sb + snq*t[ldir]
		t[ldir] = -snq*sb + csq*t[ldir]
		sb = t[1]
		t[1] = csq*sb + snq*t[ldir+1]
		t[ldir+1] = -snq*sb + csq*t[ldir+1]

		// Check if swap succeeded (s[ldir] should be small).
		if math.Abs(s[ldir]) > thresh {
			ok = false
			return ok
		}
		s[ldir] = 0

		// Zero out B's subdiagonal if small.
		if math.Abs(t[ldir]) <= thresh {
			t[ldir] = 0
		}

		// Build transformation matrices.
		li[0] = csq
		li[1] = -snq
		li[ldir] = snq
		li[ldir+1] = csq

		ir[0] = csz
		ir[1] = -snz
		ir[ldir] = snz
		ir[ldir+1] = csz

		weak = true
		strong = true
	} else {
		// Swap with at least one 2×2 block (m >= 3).
		// Follow LAPACK's approach: solve Sylvester equation, then use
		// QR and RQ factorizations to orthogonalize the transformation matrices.

		// Reset LI and IR to zero (they were initialized to identity earlier).
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				li[i*ldir+j] = 0
				ir[i*ldir+j] = 0
			}
		}

		// Solve the Sylvester equation:
		//   A11*R - L*A22 = scale*C
		//   B11*R - L*B22 = scale*F
		// where C and F are the off-diagonal blocks.
		// R and L are n1 x n2 matrices.

		// Store the solution in IR and LI:
		// IR will have R in positions (n2:m, n1:m) = (n2:n2+n1, n1:n1+n2)
		// LI will have L in positions (0:n1, 0:n2)

		var iwork [8]int
		var dsum, dscale float64

		// Call Dtgsy2 with the correct parameter order (m=n1, n=n2).
		// LAPACK stores the off-diagonal blocks in specific locations of IR and LI.
		// For simplicity, we'll copy them to temporary storage first.

		// Copy off-diagonal blocks C and F (n1 x n2) from S and T.
		// C = S[0:n1, n1:n1+n2], F = T[0:n1, n1:n1+n2]
		// Stored in row-major with leading dimension n2.
		ldcf := max(1, n2)
		var cc [4]float64 // n1 x n2 stored row-major with ldc=n2
		var ff [4]float64
		for i := 0; i < n1; i++ {
			for j := 0; j < n2; j++ {
				cc[i*ldcf+j] = s[i*ldir+n1+j]
				ff[i*ldcf+j] = t[i*ldir+n1+j]
			}
		}

		// Extract diagonal blocks.
		var a11, a22, b11, b22 [4]float64
		for i := 0; i < n1; i++ {
			for j := 0; j < n1; j++ {
				a11[i*n1+j] = s[i*ldir+j]
				b11[i*n1+j] = t[i*ldir+j]
			}
		}
		for i := 0; i < n2; i++ {
			for j := 0; j < n2; j++ {
				a22[i*n2+j] = s[(n1+i)*ldir+n1+j]
				b22[i*n2+j] = t[(n1+i)*ldir+n1+j]
			}
		}

		// Call Dtgsy2: A11*R - L*A22 = scale*C, B11*R - L*B22 = scale*F.
		// R and L are n1 x n2.
		// In row-major storage, ldc must be >= n2 (number of columns of C).
		scale, dsum, dscale, _, lok := impl.Dtgsy2(blas.NoTrans, 0, n1, n2,
			a11[:], max(1, n1), a22[:], max(1, n2), cc[:], max(1, n2),
			b11[:], max(1, n1), b22[:], max(1, n2), ff[:], max(1, n2),
			0, 1, iwork[:])
		if !lok {
			ok = false
			return ok
		}
		_ = dsum
		_ = dscale

		// cc now contains R (n1 x n2), ff contains L (n1 x n2), stored row-major with ldc=ldcf.
		// Build LI = [[-L], [scale*I_n2]] as an M x N2 matrix, then QR factorize.
		// Build IR = [[scale*I_n1, R]] in rows n2:m, then RQ factorize.

		// LI construction (M x N2): first n1 rows = -L, next n2 rows = scale*I_n2.
		for i := 0; i < n1; i++ {
			for j := 0; j < n2; j++ {
				li[i*ldir+j] = -ff[i*ldcf+j] / scale
			}
		}
		for i := 0; i < n2; i++ {
			li[(n1+i)*ldir+i] = 1
		}

		// IR construction: LAPACK puts [I_n1, R] at rows n2:m.
		// Row n2+i has: I_n1 at column i, R at columns n1:m.
		// For n1=1, n2=2: row 2 is [1, R[0,0], R[0,1]].
		// For n1=2, n2=1: row 1 is [1, 0, R[0,0]], row 2 is [0, 1, R[1,0]].
		for i := 0; i < n1; i++ {
			ir[(n2+i)*ldir+i] = 1
			for j := 0; j < n2; j++ {
				ir[(n2+i)*ldir+n1+j] = cc[i*ldcf+j] / scale
			}
		}

		// QR factorize LI (M x N2) to get orthogonal Q (M x M).
		var tau [2]float64
		impl.Dgeqr2(m, n2, li[:], ldir, tau[:n2], work)

		// Generate Q from QR factorization.
		impl.Dorg2r(m, m, n2, li[:], ldir, tau[:n2], work)

		// RQ factorize IR starting at row n2. This is n1 x m.
		var taur [2]float64
		impl.Dgerq2(n1, m, ir[n2*ldir:], ldir, taur[:n1], work)

		// Generate the orthogonal Q from RQ.
		impl.Dorgr2(m, m, n1, ir[:], ldir, taur[:n1], work)

		// Now LI and IR are orthogonal matrices.
		// Apply transformations: S_new = LI^T * S * IR^T, T_new = LI^T * T * IR^T.
		// For consistency with A = Q^T * S * Z, we need Z := Z * IR^T (not Z * IR).
		// This is handled by transposing IR in place after the local transform.
		var tmp1 [ldir * ldir]float64
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				sum = 0
				for k := 0; k < m; k++ {
					sum += s[i*ldir+k] * ir[j*ldir+k] // S * IR^T
				}
				tmp1[i*ldir+j] = sum
			}
		}
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				sum = 0
				for k := 0; k < m; k++ {
					sum += li[k*ldir+i] * tmp1[k*ldir+j]
				}
				s[i*ldir+j] = sum
			}
		}

		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				sum = 0
				for k := 0; k < m; k++ {
					sum += t[i*ldir+k] * ir[j*ldir+k] // T * IR^T
				}
				tmp1[i*ldir+j] = sum
			}
		}
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				sum = 0
				for k := 0; k < m; k++ {
					sum += li[k*ldir+i] * tmp1[k*ldir+j]
				}
				t[i*ldir+j] = sum
			}
		}

		// Transpose IR in place so Z updates use IR^T.
		for i := 0; i < m; i++ {
			for j := i + 1; j < m; j++ {
				ir[i*ldir+j], ir[j*ldir+i] = ir[j*ldir+i], ir[i*ldir+j]
			}
		}

		// Check if swap succeeded (lower-left block should be small).
		sum = 0
		for i := n2; i < m; i++ {
			for j := 0; j < n2; j++ {
				sum += math.Abs(s[i*ldir+j])
			}
		}
		if sum <= thresh*float64(m) {
			weak = true
		}
		if !weak {
			ok = false
			return ok
		}

		// Zero out small elements in lower-left.
		for i := n2; i < m; i++ {
			for j := 0; j < n2; j++ {
				s[i*ldir+j] = 0
			}
		}
		for i := 1; i < m; i++ {
			if math.Abs(t[i*ldir+i-1]) <= thresh {
				t[i*ldir+i-1] = 0
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
	// The transformation is A_new = LI^T * A_old * IR (for m=2) or A_new = LI^T * A_old * IR^T (for m>=3).
	// After the m>=3 case, IR has been transposed so IR now holds IR_orig^T.

	if j1 > 0 {
		// Update columns 0:j1-1 (left of swap region) for rows j1:j1+m.
		// A[j1:j1+m, 0:j1] = LI^T * A[j1:j1+m, 0:j1]
		bi.Dgemm(blas.Trans, blas.NoTrans, m, j1, m, 1, li[:], ldir, a[j1*lda:], lda, 0, work, j1)
		impl.Dlacpy(blas.All, m, j1, work, j1, a[j1*lda:], lda)
		bi.Dgemm(blas.Trans, blas.NoTrans, m, j1, m, 1, li[:], ldir, b[j1*ldb:], ldb, 0, work, j1)
		impl.Dlacpy(blas.All, m, j1, work, j1, b[j1*ldb:], ldb)

		// Update rows 0:j1-1 (above swap region) for columns j1:j1+m.
		// A[0:j1, j1:j1+m] = A[0:j1, j1:j1+m] * IR
		bi.Dgemm(blas.NoTrans, blas.NoTrans, j1, m, m, 1, a[j1:], lda, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, j1, m, work, m, a[j1:], lda)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, j1, m, m, 1, b[j1:], ldb, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, j1, m, work, m, b[j1:], ldb)
	}

	if j1+m < n {
		// Update rows j1+m:n-1 (below swap region) for columns j1:j1+m.
		// A[j1+m:n, j1:j1+m] = A[j1+m:n, j1:j1+m] * IR
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n-j1-m, m, m, 1, a[(j1+m)*lda+j1:], lda, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, n-j1-m, m, work, m, a[(j1+m)*lda+j1:], lda)
		bi.Dgemm(blas.NoTrans, blas.NoTrans, n-j1-m, m, m, 1, b[(j1+m)*ldb+j1:], ldb, ir[:], ldir, 0, work, m)
		impl.Dlacpy(blas.All, n-j1-m, m, work, m, b[(j1+m)*ldb+j1:], ldb)

		// Update columns j1+m:n-1 (right of swap region) for rows j1:j1+m.
		// A[j1:j1+m, j1+m:n] = LI^T * A[j1:j1+m, j1+m:n]
		bi.Dgemm(blas.Trans, blas.NoTrans, m, n-j1-m, m, 1, li[:], ldir, a[j1*lda+j1+m:], lda, 0, work, n-j1-m)
		impl.Dlacpy(blas.All, m, n-j1-m, work, n-j1-m, a[j1*lda+j1+m:], lda)
		bi.Dgemm(blas.Trans, blas.NoTrans, m, n-j1-m, m, 1, li[:], ldir, b[j1*ldb+j1+m:], ldb, 0, work, n-j1-m)
		impl.Dlacpy(blas.All, m, n-j1-m, work, n-j1-m, b[j1*ldb+j1+m:], ldb)
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
