// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math"

// Dlagv2 computes the generalized Schur factorization of a real 2×2 matrix
// pencil (A,B) where B is upper triangular.
//
// On entry, A and B are 2×2 matrices. On return, A and B are overwritten by
// the generalized Schur form:
//
//	(A11 A12)     (B11 B12)
//	(A21 A22)     (0   B22)
//
// where A21 is either zero (real eigenvalues) or has the form:
//
//	A21 = c * s * (alphar² + alphai²) / (scale1 * scale2)
//
// The outputs satisfy:
//
//	Q^T * A * Z = Schur form of A
//	Q^T * B * Z = upper triangular form of B
//
// where Q and Z are orthogonal matrices formed from Givens rotations:
//
//	Q = [csq -snq; snq csq]
//	Z = [csz -snz; snz csz]
//
// The eigenvalues are:
//
//	λ₁ = (alphar[0] + i*alphai[0]) / scale1
//	λ₂ = (alphar[1] + i*alphai[1]) / scale2
//
// Dlagv2 is an internal routine. It is exported for testing purposes.
func (Implementation) Dlagv2(a []float64, lda int, b []float64, ldb int) (csq, snq, csr, snr, csz, snz, scale1, scale2, alphar0, alphar1, alphai0, alphai1 float64) {
	switch {
	case lda < 2:
		panic(badLdA)
	case ldb < 2:
		panic(badLdB)
	case len(a) < lda+2:
		panic(shortA)
	case len(b) < ldb+2:
		panic(shortB)
	}

	safmin := dlamchS
	ulp := dlamchE

	// Compute the eigenvalues by solving the 2×2 generalized eigenvalue problem.
	a11 := a[0]
	a12 := a[1]
	a21 := a[lda]
	a22 := a[lda+1]

	b11 := b[0]
	b12 := b[1]
	b22 := b[ldb+1]

	// Scale A.
	anorm := math.Max(math.Abs(a11)+math.Abs(a21), math.Max(math.Abs(a12)+math.Abs(a22), safmin))
	ascale := 1 / anorm
	a11 *= ascale
	a12 *= ascale
	a21 *= ascale
	a22 *= ascale

	// Scale B.
	bnorm := math.Max(math.Abs(b11), math.Max(math.Abs(b12)+math.Abs(b22), safmin))
	bscale := 1 / bnorm
	b11 *= bscale
	b12 *= bscale
	b22 *= bscale

	// Check if A21 is negligible.
	if math.Abs(a21) <= ulp {
		// Matrix is already upper triangular. No transformation needed.
		// Just write back the scaled values and return identity transformation.
		a[0] = a11 * anorm
		a[1] = a12 * anorm
		a[lda] = 0
		a[lda+1] = a22 * anorm
		b[0] = b11 * bnorm
		b[1] = b12 * bnorm
		b[ldb+1] = b22 * bnorm

		csq = 1
		snq = 0
		csr = 1
		snr = 0
		csz = 1
		snz = 0

		// Compute eigenvalues.
		alphar0 = a[0]
		alphar1 = a[lda+1]
		alphai0 = 0
		alphai1 = 0
		scale1 = b[0]
		scale2 = b[ldb+1]
		return
	} else if math.Abs(b11) <= ulp {
		// B11 is negligible.
		csq, snq, _ = Implementation{}.Dlartg(a11, a21)
		a[0] = csq*a11 + snq*a21
		a[1] = csq*a12 + snq*a22
		a[lda] = -snq*a11 + csq*a21
		a[lda+1] = -snq*a12 + csq*a22
		b[0] = csq * b11
		if b[0] < 0 {
			b[0] = -b[0]
			csq = -csq
			snq = -snq
		}
		b[ldb+1] = csq*b12 + snq*b22

		csr = 1
		snr = 0
		csz = 1
		snz = 0
	} else if math.Abs(b22) <= ulp {
		// B22 is negligible.
		csz, snz, _ = Implementation{}.Dlartg(a22, -a21)
		a[0] = csz*a11 + snz*a12
		a[1] = -snz*a11 + csz*a12
		a[lda] = csz*a21 + snz*a22
		a[lda+1] = -snz*a21 + csz*a22
		b[0] = csz*b11 + snz*b12
		b[1] = -snz*b11 + csz*b12
		b[ldb+1] = csz * b22

		csq = 1
		snq = 0
		csr = 1
		snr = 0
	} else if math.Abs(b11)*math.Abs(a22)-math.Abs(a11)*math.Abs(b22) <= ulp*math.Abs(b22)*math.Abs(a21) {
		// Eigenvalues are close or one is infinite.
		csq, snq, _ = Implementation{}.Dlartg(a22*b11-a11*b22, a21*b22)
		t := csq*a11 + snq*a21
		a[lda] = -snq*a11 + csq*a21
		a[0] = t
		t = csq*a12 + snq*a22
		a[lda+1] = -snq*a12 + csq*a22
		a[1] = t
		b[0] = csq * b11
		t = csq*b12 + snq*b22
		b[ldb+1] = -snq*b12 + csq*b22
		b[1] = t

		csr = 1
		snr = 0
		csz = 1
		snz = 0
	} else {
		// General case: use full Schur decomposition.
		// Compute shifts from the pencil (A - λB).
		qq := a11/b11 - (a12/b22)*(b12/b11)
		pp := 0.5 * (a22/b22 - qq)
		rr := (a21/b22)*(a12/b11) - (a11/b11-a22/b22)*(a21/b22)*(b12/(b11*b22))
		dd := pp*pp + rr
		if dd >= 0 {
			// Real eigenvalues.
			// Compute tan(theta) for larger eigenvalue.
			dd = math.Sqrt(dd)
			if pp < 0 {
				dd = -dd
			}
			l := pp + dd
			// Compute shift to annihilate a21.
			csq, snq, _ = Implementation{}.Dlartg(l*b11-a11, a21)
			t := csq*a11 + snq*a21
			a[lda] = -snq*a11 + csq*a21
			a[0] = t
			t = csq*a12 + snq*a22
			a[lda+1] = -snq*a12 + csq*a22
			a[1] = t
			b[0] = csq * b11
			t = csq*b12 + snq*b22
			b[ldb+1] = -snq*b12 + csq*b22
			b[1] = t

			// Zero out a[lda] if it became small.
			if math.Abs(a[lda]) <= ulp*math.Max(math.Abs(a[0]), math.Abs(a[lda+1])) {
				a[lda] = 0
			}

			csr = 1
			snr = 0
			csz = 1
			snz = 0
		} else {
			// Complex eigenvalues.
			// First apply Q to make B upper triangular.
			csq, snq, _ = Implementation{}.Dlartg(b11, 0)

			t := csq*a11 + snq*a21
			a[lda] = -snq*a11 + csq*a21
			a[0] = t
			t = csq*a12 + snq*a22
			a[lda+1] = -snq*a12 + csq*a22
			a[1] = t
			b[0] = csq * b11
			t = csq*b12 + snq*b22
			b[ldb+1] = -snq*b12 + csq*b22
			b[1] = t

			// Now apply Z to annihilate b[1].
			csz, snz, _ = Implementation{}.Dlartg(b[ldb+1], -b[1])
			t = csz*a[0] + snz*a[1]
			a[1] = -snz*a[0] + csz*a[1]
			a[0] = t
			t = csz*a[lda] + snz*a[lda+1]
			a[lda+1] = -snz*a[lda] + csz*a[lda+1]
			a[lda] = t
			t = csz*b[0] + snz*b[1]
			b[1] = 0
			b[0] = t
			b[ldb+1] = csz*b[ldb+1] - snz*0

			csr = 1
			snr = 0
		}
	}

	// Unscale.
	a[0] *= anorm
	a[1] *= anorm
	a[lda] *= anorm
	a[lda+1] *= anorm
	b[0] *= bnorm
	b[1] *= bnorm
	b[ldb+1] *= bnorm

	// Compute eigenvalue representation.
	// For real eigenvalues:
	if a[lda] == 0 {
		alphar0 = a[0]
		alphar1 = a[lda+1]
		alphai0 = 0
		alphai1 = 0
		scale1 = b[0]
		scale2 = b[ldb+1]
	} else {
		// Complex eigenvalues.
		alphar0 = a[0]
		alphar1 = a[0]
		alphai0 = math.Sqrt(math.Abs(a[1])) * math.Sqrt(math.Abs(a[lda]))
		alphai1 = -alphai0
		scale1 = b[0]
		scale2 = b[0]
	}

	return
}
