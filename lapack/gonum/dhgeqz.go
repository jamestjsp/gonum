// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

// Dhgeqz computes the eigenvalues of a real matrix pair (H,T) where H is upper
// Hessenberg and T is upper triangular, using the double-shift QZ method.
//
// Dhgeqz also computes the generalized Schur form:
//
//	Q^T * H * Z = S  (upper quasi-triangular)
//	Q^T * T * Z = P  (upper triangular)
//
// Optionally, it also computes the orthogonal matrices Q and Z.
//
// job specifies what is computed:
//
//	lapack.EigenvaluesOnly: only eigenvalues
//	lapack.EigenvaluesAndSchur: eigenvalues and Schur form
//
// compq and compz specify how Q and Z are computed:
//
//	lapack.SchurNone: no update
//	lapack.SchurHess: initialize to identity and update
//	lapack.SchurOrig: update the input matrices
//
// n is the order of the matrices. ilo and ihi specify the working submatrix
// computed by a previous call to Dggbal.
//
// On entry, H is an n×n upper Hessenberg matrix and T is an n×n upper
// triangular matrix. On exit, if job is lapack.EigenvaluesAndSchur, H is
// overwritten by S and T is overwritten by P.
//
// alphar, alphai, and beta must have length n. On exit, the eigenvalues are
// (alphar[j] + i*alphai[j]) / beta[j] for j = 0,...,n-1.
//
// work must have length at least lwork. If lwork is -1, workspace query is
// performed and optimal size is returned in work[0].
//
// Dhgeqz returns ok=false if the iteration fails to converge.
//
// Dhgeqz is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dhgeqz(job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi int,
	h []float64, ldh int, t []float64, ldt int, alphar, alphai, beta []float64,
	q []float64, ldq int, z []float64, ldz int, work []float64, lwork int) (ok bool) {

	ilschr := job == lapack.EigenvaluesAndSchur
	ilq := compq == lapack.SchurHess || compq == lapack.SchurOrig
	ilz := compz == lapack.SchurHess || compz == lapack.SchurOrig

	switch {
	case job != lapack.EigenvaluesOnly && job != lapack.EigenvaluesAndSchur:
		panic(badSchurJob)
	case compq != lapack.SchurNone && compq != lapack.SchurHess && compq != lapack.SchurOrig:
		panic(badSchurComp)
	case compz != lapack.SchurNone && compz != lapack.SchurHess && compz != lapack.SchurOrig:
		panic(badSchurComp)
	case n < 0:
		panic(nLT0)
	case ilo < 0 || max(0, n-1) < ilo:
		panic(badIlo)
	case ihi < min(ilo, n-1) || n <= ihi:
		panic(badIhi)
	case ldh < max(1, n):
		panic(badLdH)
	case ldt < max(1, n):
		panic(badLdT)
	case ldq < 1, ilq && ldq < n:
		panic(badLdQ)
	case ldz < 1, ilz && ldz < n:
		panic(badLdZ)
	case lwork < max(1, n) && lwork != -1:
		panic(badLWork)
	}

	// Workspace query.
	if lwork == -1 {
		work[0] = float64(max(1, n))
		return true
	}

	if n == 0 {
		work[0] = 1
		return true
	}

	switch {
	case len(h) < (n-1)*ldh+n:
		panic(shortH)
	case len(t) < (n-1)*ldt+n:
		panic(shortT)
	case len(alphar) < n:
		panic("lapack: insufficient length of alphar")
	case len(alphai) < n:
		panic("lapack: insufficient length of alphai")
	case len(beta) < n:
		panic("lapack: insufficient length of beta")
	case ilq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case ilz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(work) < lwork:
		panic(shortWork)
	}

	work[0] = float64(n)

	// Initialize Q and Z to identity if requested.
	if compq == lapack.SchurHess {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}
	if compz == lapack.SchurHess {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz)
	}

	// Machine constants.
	eps := dlamchE
	safmin := dlamchS
	ulp := eps
	anorm := impl.Dlanhs(lapack.Frobenius, n, h, ldh, nil)
	bnorm := impl.Dlantr(lapack.Frobenius, blas.Upper, blas.NonUnit, n, n, t, ldt, nil)
	atol := math.Max(safmin, ulp*anorm)
	btol := math.Max(safmin, ulp*bnorm)
	ascale := 1 / math.Max(safmin, anorm)
	bscale := 1 / math.Max(safmin, bnorm)

	// Set eigenvalues for rows ilo-1:0 and ihi+1:n.
	for j := 0; j < ilo; j++ {
		alphar[j] = h[j*ldh+j]
		alphai[j] = 0
		beta[j] = t[j*ldt+j]
	}
	for j := ihi + 1; j < n; j++ {
		alphar[j] = h[j*ldh+j]
		alphai[j] = 0
		beta[j] = t[j*ldt+j]
	}

	// Main QZ iteration loop.
	ifirst := ilo
	ilast := ihi
	ifrstm := ilo
	ilastm := ihi
	if ilschr {
		ifrstm = 0
		ilastm = n - 1
	}

	const maxit = 30
	for jiter := 0; jiter < maxit*(ihi-ilo+1); jiter++ {
		// Check for deflation.
		if ilast < ilo {
			break
		}

		// Check for negligible subdiagonal element.
		for j := ilast; j > ilo; j-- {
			if math.Abs(h[j*ldh+j-1]) <= atol {
				h[j*ldh+j-1] = 0
				ifirst = j
				goto L60
			}
			tst1 := math.Abs(h[(j-1)*ldh+j-1]) + math.Abs(h[j*ldh+j])
			if tst1 == 0 {
				if j >= ilo+2 {
					tst1 += math.Abs(h[(j-1)*ldh+j-2])
				}
				if j < ilast {
					tst1 += math.Abs(h[(j+1)*ldh+j])
				}
			}
			if math.Abs(h[j*ldh+j-1]) <= ulp*tst1 {
				temp := math.Max(math.Abs(h[j*ldh+j-1]), math.Abs(h[(j-1)*ldh+j]))
				temp2 := math.Min(math.Abs(h[j*ldh+j-1]), math.Abs(h[(j-1)*ldh+j]))
				temp3 := math.Max(math.Abs(h[j*ldh+j]), math.Abs(h[(j-1)*ldh+j-1]-h[j*ldh+j]))
				temp4 := math.Min(math.Abs(h[j*ldh+j]), math.Abs(h[(j-1)*ldh+j-1]-h[j*ldh+j]))
				if temp*temp2/math.Max(safmin, temp3*temp4) <= math.Max(safmin, ulp*temp3) {
					h[j*ldh+j-1] = 0
					ifirst = j
					goto L60
				}
			}
		}
		ifirst = ilo

	L60:
		// Check for negligible off-diagonal element in T.
		for j := ilast; j >= ifirst+1; j-- {
			if math.Abs(t[j*ldt+j-1]) <= btol {
				t[j*ldt+j-1] = 0
			}
		}

		// Check for deflation.
		if ifirst == ilast {
			// Single eigenvalue.
			alphar[ilast] = h[ilast*ldh+ilast]
			alphai[ilast] = 0
			beta[ilast] = t[ilast*ldt+ilast]
			ilast--
			continue
		}

		if ifirst == ilast-1 {
			// 2x2 block.
			// Compute eigenvalues of 2x2 block.
			var s1, s2, wr1, wr2, wi float64
			s1, s2, wr1, wr2, wi = impl.Dlag2(h[ifirst*ldh+ifirst:], ldh, t[ifirst*ldt+ifirst:], ldt)
			if wi == 0 {
				// Real eigenvalues.
				alphar[ifirst] = wr1
				alphai[ifirst] = 0
				beta[ifirst] = s1
				alphar[ilast] = wr2
				alphai[ilast] = 0
				beta[ilast] = s2
			} else {
				// Complex conjugate pair.
				alphar[ifirst] = wr1
				alphai[ifirst] = wi
				beta[ifirst] = s1
				alphar[ilast] = wr1
				alphai[ilast] = -wi
				beta[ilast] = s1
			}
			ilast -= 2
			continue
		}

		// Check for infinite eigenvalue.
		if t[ilast*ldt+ilast] == 0 {
			// Eigenvalue at infinity.
			alphar[ilast] = h[ilast*ldh+ilast]
			alphai[ilast] = 0
			beta[ilast] = 0
			ilast--
			continue
		}

		// Do one QZ step.
		ok = impl.doQZStep(ilschr, ilq, ilz, n, ifirst, ilast, ifrstm, ilastm,
			h, ldh, t, ldt, q, ldq, z, ldz, ascale, bscale, safmin)
		if !ok {
			// Try exceptional shift.
			for j := ilast - 1; j >= ifirst; j-- {
				temp := 0.0
				if j == ifirst {
					temp = math.Abs(h[(j+1)*ldh+j])
				} else if j == ilast-1 {
					temp = math.Abs(h[j*ldh+j-1])
				} else {
					temp = math.Abs(h[(j+1)*ldh+j]) + math.Abs(h[j*ldh+j-1])
				}
				if math.Abs(h[j*ldh+j]) <= ulp*temp {
					h[j*ldh+j] = 0
				}
			}
		}
	}

	// Set remaining eigenvalues.
	if ilast >= ilo {
		// Iteration did not fully converge.
		return false
	}

	return true
}

// doQZStep performs one step of the QZ algorithm.
func (impl Implementation) doQZStep(ilschr, ilq, ilz bool, n, ifirst, ilast, ifrstm, ilastm int,
	h []float64, ldh int, t []float64, ldt int, q []float64, ldq int, z []float64, ldz int,
	ascale, bscale, safmin float64) bool {

	// Compute shift.
	a11 := ascale * h[ifirst*ldh+ifirst]
	a21 := ascale * h[(ifirst+1)*ldh+ifirst]
	a22 := ascale * h[(ifirst+1)*ldh+ifirst+1]
	b11 := bscale * t[ifirst*ldt+ifirst]
	b22 := bscale * t[(ifirst+1)*ldt+ifirst+1]

	// Calculate shift.
	s1 := a11*b22 - a22*b11
	s2 := a21 * b22
	if math.Abs(s2) <= safmin {
		return true // Already converged.
	}

	// Apply transformations.
	for j := ifirst; j < ilast; j++ {
		if j > ifirst {
			a21 = h[(j+1)*ldh+j-1]
		}

		// Compute Givens rotation to annihilate a21.
		var cs, sn, r float64
		if j == ifirst {
			cs, sn, r = impl.Dlartg(s1, s2)
		} else {
			cs, sn, r = impl.Dlartg(h[j*ldh+j-1], a21)
			h[j*ldh+j-1] = r
			h[(j+1)*ldh+j-1] = 0
		}

		// Apply rotation from left to H and T.
		for jc := j; jc <= ilastm; jc++ {
			temp := cs*h[j*ldh+jc] + sn*h[(j+1)*ldh+jc]
			h[(j+1)*ldh+jc] = -sn*h[j*ldh+jc] + cs*h[(j+1)*ldh+jc]
			h[j*ldh+jc] = temp
		}
		for jc := ifrstm; jc <= min(j+2, ilastm); jc++ {
			temp := cs*t[j*ldt+jc] + sn*t[(j+1)*ldt+jc]
			t[(j+1)*ldt+jc] = -sn*t[j*ldt+jc] + cs*t[(j+1)*ldt+jc]
			t[j*ldt+jc] = temp
		}
		if ilq {
			for jr := 0; jr < n; jr++ {
				temp := cs*q[jr*ldq+j] + sn*q[jr*ldq+j+1]
				q[jr*ldq+j+1] = -sn*q[jr*ldq+j] + cs*q[jr*ldq+j+1]
				q[jr*ldq+j] = temp
			}
		}

		// Annihilate t[j+1,j].
		if t[(j+1)*ldt+j] != 0 {
			cs, sn, r = impl.Dlartg(t[(j+1)*ldt+j+1], t[(j+1)*ldt+j])
			t[(j+1)*ldt+j+1] = r
			t[(j+1)*ldt+j] = 0

			// Apply rotation from right to H and T.
			for jr := ifrstm; jr <= min(j+2, ilast); jr++ {
				temp := cs*h[jr*ldh+j+1] + sn*h[jr*ldh+j]
				h[jr*ldh+j] = -sn*h[jr*ldh+j+1] + cs*h[jr*ldh+j]
				h[jr*ldh+j+1] = temp
			}
			for jr := ifrstm; jr <= j; jr++ {
				temp := cs*t[jr*ldt+j+1] + sn*t[jr*ldt+j]
				t[jr*ldt+j] = -sn*t[jr*ldt+j+1] + cs*t[jr*ldt+j]
				t[jr*ldt+j+1] = temp
			}
			if ilz {
				for jr := 0; jr < n; jr++ {
					temp := cs*z[jr*ldz+j+1] + sn*z[jr*ldz+j]
					z[jr*ldz+j] = -sn*z[jr*ldz+j+1] + cs*z[jr*ldz+j]
					z[jr*ldz+j+1] = temp
				}
			}
		}
	}

	return true
}
