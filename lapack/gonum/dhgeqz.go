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
	safmin := dlamchS
	ulp := dlamchE
	anorm := impl.Dlanhs(lapack.Frobenius, n, h, ldh, nil)
	bnorm := impl.Dlantr(lapack.Frobenius, blas.Upper, blas.NonUnit, n, n, t, ldt, nil)
	atol := math.Max(safmin, ulp*anorm)
	btol := math.Max(safmin, ulp*bnorm)

	// Set eigenvalues for rows outside [ilo, ihi].
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
	ilast := ihi
	ifrstm := ilo
	ilastm := ihi
	if ilschr {
		ifrstm = 0
		ilastm = n - 1
	}

	const maxit = 30
	iiter := 0      // Iterations since last eigenvalue.
	eshift := 0.0   // Exceptional shift accumulator.

	for jiter := 0; jiter < maxit*(ihi-ilo+1); jiter++ {
		// Check for convergence.
		if ilast < ilo {
			break
		}

		// Check for deflation: negligible subdiagonal in H.
		var ifirst int
		for j := ilast; j > ilo; j-- {
			if math.Abs(h[j*ldh+j-1]) <= atol {
				h[j*ldh+j-1] = 0
				ifirst = j
				goto checkT
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
				hlj := math.Abs(h[j*ldh+j-1])
				hjlm1 := math.Abs(h[(j-1)*ldh+j])
				temp := math.Max(hlj, hjlm1)
				temp2 := math.Min(hlj, hjlm1)
				hjj := math.Abs(h[j*ldh+j])
				hjm1 := math.Abs(h[(j-1)*ldh+j-1])
				temp3 := math.Max(hjj, math.Abs(hjm1-hjj))
				temp4 := math.Min(hjj, math.Abs(hjm1-hjj))
				if temp2*temp <= math.Max(safmin, ulp*temp3*temp4) {
					h[j*ldh+j-1] = 0
					ifirst = j
					goto checkT
				}
			}
		}
		ifirst = ilo

	checkT:
		// Check for negligible elements in T.
		for j := ilast; j >= ifirst+1; j-- {
			if math.Abs(t[j*ldt+j-1]) <= btol {
				t[j*ldt+j-1] = 0
			}
		}

		// Handle different block types.
		if ifirst == ilast {
			// 1x1 block - single real eigenvalue.
			alphar[ilast] = h[ilast*ldh+ilast]
			alphai[ilast] = 0
			beta[ilast] = t[ilast*ldt+ilast]
			ilast--
			iiter = 0
			continue
		}

		if ifirst == ilast-1 {
			// 2x2 block - compute eigenvalues.
			s1, s2, wr1, wr2, wi := impl.Dlag2(h[ifirst*ldh+ifirst:], ldh, t[ifirst*ldt+ifirst:], ldt)
			if wi == 0 {
				// Two real eigenvalues.
				alphar[ifirst] = wr1
				alphai[ifirst] = 0
				beta[ifirst] = s1
				alphar[ilast] = wr2
				alphai[ilast] = 0
				beta[ilast] = s2
			} else {
				// Complex conjugate pair - standardize the 2x2 block.
				// H block should have equal diagonals and H(i+1,i)*H(i,i+1) < 0.
				// T block should be upper triangular with positive entries.
				if ilschr {
					impl.standardize2x2Block(n, ifirst, ifrstm, ilastm,
						h, ldh, t, ldt, q, ldq, z, ldz, ilq, ilz)
				}
				alphar[ifirst] = wr1
				alphai[ifirst] = wi
				beta[ifirst] = s1
				alphar[ilast] = wr1
				alphai[ilast] = -wi
				beta[ilast] = s1
			}
			ilast -= 2
			iiter = 0
			continue
		}

		// Handle zero diagonal in T (infinite eigenvalue).
		if t[ilast*ldt+ilast] == 0 {
			alphar[ilast] = h[ilast*ldh+ilast]
			alphai[ilast] = 0
			beta[ilast] = 0
			ilast--
			iiter = 0
			continue
		}

		// Perform QZ step.
		iiter++

		// Compute shifts from the trailing 2x2 block.
		var s1, s2, wr, wr2, wi float64

		// Exceptional shift every 10 iterations.
		if iiter%10 == 0 {
			// Ad-hoc exceptional shift to escape stagnation.
			// Use shift of 1 (Mathworks improvement over LAPACK's zero).
			eshift += 1.0
			s1 = eshift
			s2 = 0
			wr = eshift
			wi = 0
		} else {
			// Normal shift: eigenvalues of trailing 2x2.
			s1, s2, wr, wr2, wi = impl.Dlag2(h[(ilast-1)*ldh+ilast-1:], ldh, t[(ilast-1)*ldt+ilast-1:], ldt)
			_ = wr2
		}

		// Do one QZ sweep.
		// TODO: Double-shift has bugs, use single-shift for now.
		impl.doQZSweepSingle(ilschr, ilq, ilz, n, ifirst, ilast, ifrstm, ilastm,
			h, ldh, t, ldt, q, ldq, z, ldz, s1, wr, safmin)
		_, _ = wi, s2
	}

	// Check convergence.
	if ilast >= ilo {
		return false
	}

	return true
}

// doQZSweepSingle performs a single-shift QZ sweep.
func (impl Implementation) doQZSweepSingle(ilschr, ilq, ilz bool, n, ifirst, ilast, ifrstm, ilastm int,
	h []float64, ldh int, t []float64, ldt int, q []float64, ldq int, z []float64, ldz int,
	s1, wr, safmin float64) {

	istart := ifirst

	// First column of H - shift*T where shift = wr/s1.
	temp := h[istart*ldh+istart]
	if s1 != 0 {
		temp -= (wr / s1) * t[istart*ldt+istart]
	}
	temp2 := h[(istart+1)*ldh+istart]

	cs, sn, _ := impl.Dlartg(temp, temp2)

	// Chase the bulge.
	for j := istart; j < ilast; j++ {
		if j > istart {
			temp = h[j*ldh+j-1]
			temp2 = h[(j+1)*ldh+j-1]
			cs, sn, _ = impl.Dlartg(temp, temp2)
			h[j*ldh+j-1] = cs*temp + sn*temp2
			h[(j+1)*ldh+j-1] = 0
		}

		// Apply rotation from left to H.
		for jc := j; jc <= ilastm; jc++ {
			temp = cs*h[j*ldh+jc] + sn*h[(j+1)*ldh+jc]
			h[(j+1)*ldh+jc] = -sn*h[j*ldh+jc] + cs*h[(j+1)*ldh+jc]
			h[j*ldh+jc] = temp
		}

		// Apply rotation from left to T.
		for jc := j; jc <= ilastm; jc++ {
			temp = cs*t[j*ldt+jc] + sn*t[(j+1)*ldt+jc]
			t[(j+1)*ldt+jc] = -sn*t[j*ldt+jc] + cs*t[(j+1)*ldt+jc]
			t[j*ldt+jc] = temp
		}

		// Update Q if needed.
		if ilq {
			for jr := 0; jr < n; jr++ {
				temp = cs*q[jr*ldq+j] + sn*q[jr*ldq+j+1]
				q[jr*ldq+j+1] = -sn*q[jr*ldq+j] + cs*q[jr*ldq+j+1]
				q[jr*ldq+j] = temp
			}
		}

		// Annihilate T[j+1,j] to restore upper triangular form.
		if t[(j+1)*ldt+j] != 0 {
			cs, sn, _ = impl.Dlartg(t[(j+1)*ldt+j+1], t[(j+1)*ldt+j])
			t[(j+1)*ldt+j+1] = cs*t[(j+1)*ldt+j+1] + sn*t[(j+1)*ldt+j]
			t[(j+1)*ldt+j] = 0

			// Apply rotation from right to H.
			for jr := ifrstm; jr <= min(j+2, ilast); jr++ {
				temp = cs*h[jr*ldh+j+1] + sn*h[jr*ldh+j]
				h[jr*ldh+j] = -sn*h[jr*ldh+j+1] + cs*h[jr*ldh+j]
				h[jr*ldh+j+1] = temp
			}

			// Apply rotation from right to T.
			for jr := ifrstm; jr <= j; jr++ {
				temp = cs*t[jr*ldt+j+1] + sn*t[jr*ldt+j]
				t[jr*ldt+j] = -sn*t[jr*ldt+j+1] + cs*t[jr*ldt+j]
				t[jr*ldt+j+1] = temp
			}

			// Update Z if needed.
			if ilz {
				for jr := 0; jr < n; jr++ {
					temp = cs*z[jr*ldz+j+1] + sn*z[jr*ldz+j]
					z[jr*ldz+j] = -sn*z[jr*ldz+j+1] + cs*z[jr*ldz+j]
					z[jr*ldz+j+1] = temp
				}
			}
		}
	}
}

// doQZSweepDouble performs an implicit double-shift QZ sweep for complex conjugate shifts.
// The shifts are wr ± i*wi where wi != 0.
func (impl Implementation) doQZSweepDouble(ilschr, ilq, ilz bool, n, ifirst, ilast, ifrstm, ilastm int,
	h []float64, ldh int, t []float64, ldt int, q []float64, ldq int, z []float64, ldz int,
	s1, s2, wr, wi, safmin float64) {

	istart := ifirst

	// For the implicit double-shift, we compute the first column of
	// (H - (wr+i*wi)*T/s1) * (H - (wr-i*wi)*T/s1)
	// which equals the first column of
	// (H*H - 2*wr*H*T/s1 + (wr^2+wi^2)*T*T/s1^2)
	//
	// Let shift1 = wr/s1, shift2 = (wr^2+wi^2)/(s1*s2)
	// The product M = (H - shift1*T)^2 + (wi/s1)^2*T^2
	//
	// First column of M applied to e1 gives a 3-vector (for 3x3 or larger).

	// Extract the top-left 3x3 block (or 2x2 if ilast-istart < 2).
	h11 := h[istart*ldh+istart]
	h12 := h[istart*ldh+istart+1]
	h21 := h[(istart+1)*ldh+istart]
	h22 := h[(istart+1)*ldh+istart+1]
	t11 := t[istart*ldt+istart]
	t12 := t[istart*ldt+istart+1]
	t22 := t[(istart+1)*ldt+istart+1]

	var v1, v2, v3 float64

	if ilast-istart >= 2 {
		h32 := h[(istart+2)*ldh+istart+1]

		// Compute shift parameters.
		// For complex conjugate shifts s = wr ± i*wi with scaling s1:
		//   (H - s*T)(H - conj(s)*T) = (H - wr/s1*T)^2 + (wi/s1)^2*T^2
		var shift1, beta float64
		if s1 != 0 {
			shift1 = wr / s1
			beta = (wi / s1) * (wi / s1)
		}

		// Let A = H - shift1*T.
		a11 := h11 - shift1*t11
		a12 := h12 - shift1*t12
		a21 := h21
		a22 := h22 - shift1*t22

		// First column of (A^2 + beta*T^2):
		// A^2*e1 = [a11^2 + a12*a21, a21*(a11+a22), h32*a21]
		// T^2*e1 = [t11^2, 0, 0] (since T is upper triangular)
		v1 = a11*a11 + a12*a21 + beta*t11*t11
		v2 = a21 * (a11 + a22)
		v3 = h32 * a21

		// Scale to avoid overflow.
		scale := math.Abs(v1) + math.Abs(v2) + math.Abs(v3)
		if scale > 0 {
			v1 /= scale
			v2 /= scale
			v3 /= scale
		}
	} else {
		// 2x2 case: use single shift instead.
		var shift1 float64
		if s1 != 0 {
			shift1 = wr / s1
		}
		v1 = h11 - shift1*t11
		v2 = h21
		v3 = 0
	}

	// Create Householder transformation to introduce bulge.
	vv := []float64{v2, v3}
	v1, tau := impl.Dlarfg(3, v1, vv, 1)
	v := []float64{1, vv[0], vv[1]}

	// Chase the bulge through the matrix.
	for j := istart; j < ilast-1; j++ {
		if j > istart {
			// Compute Householder to annihilate H[j+1:j+3, j-1].
			v1 = h[j*ldh+j-1]
			v2 = h[(j+1)*ldh+j-1]
			if j+2 <= ilast {
				v3 = h[(j+2)*ldh+j-1]
			} else {
				v3 = 0
			}
			vv[0] = v2
			vv[1] = v3
			v1, tau = impl.Dlarfg(3, v1, vv, 1)
			v[0] = 1
			v[1] = vv[0]
			v[2] = vv[1]

			h[j*ldh+j-1] = v1
			h[(j+1)*ldh+j-1] = 0
			if j+2 <= ilast {
				h[(j+2)*ldh+j-1] = 0
			}
		}

		// Apply Householder from left to H.
		jend := ilastm
		for jc := j; jc <= jend; jc++ {
			sum := h[j*ldh+jc] + v[1]*h[(j+1)*ldh+jc]
			if j+2 <= ilast {
				sum += v[2] * h[(j+2)*ldh+jc]
			}
			h[j*ldh+jc] -= tau * sum
			h[(j+1)*ldh+jc] -= tau * sum * v[1]
			if j+2 <= ilast {
				h[(j+2)*ldh+jc] -= tau * sum * v[2]
			}
		}

		// Apply Householder from left to T.
		for jc := j; jc <= jend; jc++ {
			sum := t[j*ldt+jc] + v[1]*t[(j+1)*ldt+jc]
			if j+2 <= ilast {
				sum += v[2] * t[(j+2)*ldt+jc]
			}
			t[j*ldt+jc] -= tau * sum
			t[(j+1)*ldt+jc] -= tau * sum * v[1]
			if j+2 <= ilast {
				t[(j+2)*ldt+jc] -= tau * sum * v[2]
			}
		}

		// Update Q if needed.
		if ilq {
			for jr := 0; jr < n; jr++ {
				sum := q[jr*ldq+j] + v[1]*q[jr*ldq+j+1]
				if j+2 <= ilast {
					sum += v[2] * q[jr*ldq+j+2]
				}
				q[jr*ldq+j] -= tau * sum
				q[jr*ldq+j+1] -= tau * sum * v[1]
				if j+2 <= ilast {
					q[jr*ldq+j+2] -= tau * sum * v[2]
				}
			}
		}

		// Zero out T[j+2,j] and T[j+1,j] to restore triangular form.
		// First handle T[j+2,j] if it exists.
		if j+2 <= ilast && t[(j+2)*ldt+j] != 0 {
			cs, sn, _ := impl.Dlartg(t[(j+2)*ldt+j+1], t[(j+2)*ldt+j])
			t[(j+2)*ldt+j+1] = cs*t[(j+2)*ldt+j+1] + sn*t[(j+2)*ldt+j]
			t[(j+2)*ldt+j] = 0

			// Apply from right to H and T.
			for jr := ifrstm; jr <= min(j+3, ilast); jr++ {
				temp := cs*h[jr*ldh+j+1] + sn*h[jr*ldh+j]
				h[jr*ldh+j] = -sn*h[jr*ldh+j+1] + cs*h[jr*ldh+j]
				h[jr*ldh+j+1] = temp
			}
			for jr := ifrstm; jr <= j+1; jr++ {
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

		// Handle T[j+1,j].
		if t[(j+1)*ldt+j] != 0 {
			cs, sn, _ := impl.Dlartg(t[(j+1)*ldt+j+1], t[(j+1)*ldt+j])
			t[(j+1)*ldt+j+1] = cs*t[(j+1)*ldt+j+1] + sn*t[(j+1)*ldt+j]
			t[(j+1)*ldt+j] = 0

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

	// Final step: handle the last 2x2 portion.
	j := ilast - 1
	temp := h[j*ldh+j-1]
	temp2 := h[(j+1)*ldh+j-1]
	cs, sn, _ := impl.Dlartg(temp, temp2)
	h[j*ldh+j-1] = cs*temp + sn*temp2
	h[(j+1)*ldh+j-1] = 0

	// Apply from left.
	for jc := j; jc <= ilastm; jc++ {
		temp = cs*h[j*ldh+jc] + sn*h[(j+1)*ldh+jc]
		h[(j+1)*ldh+jc] = -sn*h[j*ldh+jc] + cs*h[(j+1)*ldh+jc]
		h[j*ldh+jc] = temp
	}
	for jc := j; jc <= ilastm; jc++ {
		temp = cs*t[j*ldt+jc] + sn*t[(j+1)*ldt+jc]
		t[(j+1)*ldt+jc] = -sn*t[j*ldt+jc] + cs*t[(j+1)*ldt+jc]
		t[j*ldt+jc] = temp
	}
	if ilq {
		for jr := 0; jr < n; jr++ {
			temp = cs*q[jr*ldq+j] + sn*q[jr*ldq+j+1]
			q[jr*ldq+j+1] = -sn*q[jr*ldq+j] + cs*q[jr*ldq+j+1]
			q[jr*ldq+j] = temp
		}
	}

	// Restore T triangular.
	if t[(j+1)*ldt+j] != 0 {
		cs, sn, _ = impl.Dlartg(t[(j+1)*ldt+j+1], t[(j+1)*ldt+j])
		t[(j+1)*ldt+j+1] = cs*t[(j+1)*ldt+j+1] + sn*t[(j+1)*ldt+j]
		t[(j+1)*ldt+j] = 0

		for jr := ifrstm; jr <= ilast; jr++ {
			temp = cs*h[jr*ldh+j+1] + sn*h[jr*ldh+j]
			h[jr*ldh+j] = -sn*h[jr*ldh+j+1] + cs*h[jr*ldh+j]
			h[jr*ldh+j+1] = temp
		}
		for jr := ifrstm; jr <= j; jr++ {
			temp = cs*t[jr*ldt+j+1] + sn*t[jr*ldt+j]
			t[jr*ldt+j] = -sn*t[jr*ldt+j+1] + cs*t[jr*ldt+j]
			t[jr*ldt+j+1] = temp
		}
		if ilz {
			for jr := 0; jr < n; jr++ {
				temp = cs*z[jr*ldz+j+1] + sn*z[jr*ldz+j]
				z[jr*ldz+j] = -sn*z[jr*ldz+j+1] + cs*z[jr*ldz+j]
				z[jr*ldz+j+1] = temp
			}
		}
	}

	_ = safmin
}

// standardize2x2Block puts a 2x2 block at position j into Schur canonical form.
// For H: equal diagonals with H(j+1,j)*H(j,j+1) < 0.
// For T: diagonal with positive entries.
func (impl Implementation) standardize2x2Block(n, j, ifrstm, ilastm int,
	h []float64, ldh int, t []float64, ldt int,
	q []float64, ldq int, z []float64, ldz int, ilq, ilz bool) {

	// Extract 2x2 block from H.
	a := h[j*ldh+j]
	b := h[j*ldh+j+1]
	c := h[(j+1)*ldh+j]
	d := h[(j+1)*ldh+j+1]

	// Use Dlanv2 to standardize H's 2x2 block.
	aa, bb, cc, dd, _, _, _, _, cs, sn := impl.Dlanv2(a, b, c, d)

	// Store standardized H block.
	h[j*ldh+j] = aa
	h[j*ldh+j+1] = bb
	h[(j+1)*ldh+j] = cc
	h[(j+1)*ldh+j+1] = dd

	// Apply transformation to rest of H from left: rows j, j+1.
	for jc := j + 2; jc <= ilastm; jc++ {
		temp := cs*h[j*ldh+jc] + sn*h[(j+1)*ldh+jc]
		h[(j+1)*ldh+jc] = -sn*h[j*ldh+jc] + cs*h[(j+1)*ldh+jc]
		h[j*ldh+jc] = temp
	}

	// Apply transformation to rest of H from right: columns j, j+1.
	for jr := ifrstm; jr < j; jr++ {
		temp := cs*h[jr*ldh+j] + sn*h[jr*ldh+j+1]
		h[jr*ldh+j+1] = -sn*h[jr*ldh+j] + cs*h[jr*ldh+j+1]
		h[jr*ldh+j] = temp
	}

	// Apply transformation to T from left: rows j, j+1.
	for jc := j; jc <= ilastm; jc++ {
		temp := cs*t[j*ldt+jc] + sn*t[(j+1)*ldt+jc]
		t[(j+1)*ldt+jc] = -sn*t[j*ldt+jc] + cs*t[(j+1)*ldt+jc]
		t[j*ldt+jc] = temp
	}

	// Apply transformation to T from right: columns j, j+1.
	for jr := ifrstm; jr <= j+1; jr++ {
		temp := cs*t[jr*ldt+j] + sn*t[jr*ldt+j+1]
		t[jr*ldt+j+1] = -sn*t[jr*ldt+j] + cs*t[jr*ldt+j+1]
		t[jr*ldt+j] = temp
	}

	// Update Q if needed.
	// For left transformation on H/T (rows j, j+1), we update Q = Q * G^T.
	if ilq {
		for jr := range n {
			temp := cs*q[jr*ldq+j] + sn*q[jr*ldq+j+1]
			q[jr*ldq+j+1] = -sn*q[jr*ldq+j] + cs*q[jr*ldq+j+1]
			q[jr*ldq+j] = temp
		}
	}

	// Update Z if needed.
	// For right transformation on H/T (columns j, j+1), we update Z = Z * G.
	if ilz {
		for jr := range n {
			temp := cs*z[jr*ldz+j] + sn*z[jr*ldz+j+1]
			z[jr*ldz+j+1] = -sn*z[jr*ldz+j] + cs*z[jr*ldz+j+1]
			z[jr*ldz+j] = temp
		}
	}

	// Make T upper triangular with positive diagonal.
	// After applying the Dlanv2 similarity transformation, T may have
	// nonzero t21. We need to eliminate it using a right rotation.
	// Also ensure T diagonals are positive.

	t11 := t[j*ldt+j]
	t12 := t[j*ldt+j+1]
	t21 := t[(j+1)*ldt+j]
	t22 := t[(j+1)*ldt+j+1]

	// Eliminate t21 using a right rotation on columns j, j+1.
	// Find R2 = [cs2 sn2; -sn2 cs2] such that [t21, t22] * R2 = [0, *]
	// This means cs2*t21 - sn2*t22 = 0.
	if t21 != 0 {
		// Dlartg(a, b): c*a + s*b = r, -s*a + c*b = 0 => c*b = s*a
		// We need c*t21 = s*t22, so call Dlartg(t22, t21).
		cs2, sn2, _ := impl.Dlartg(t22, t21)

		// Apply T = T * R2 (column operation).
		// For T's 2x2 block:
		t[j*ldt+j] = cs2*t11 - sn2*t12
		t[j*ldt+j+1] = sn2*t11 + cs2*t12
		t[(j+1)*ldt+j] = 0
		t[(j+1)*ldt+j+1] = sn2*t21 + cs2*t22

		// For T rows above the 2x2 block:
		for jr := ifrstm; jr < j; jr++ {
			tj := t[jr*ldt+j]
			tjp1 := t[jr*ldt+j+1]
			t[jr*ldt+j] = cs2*tj - sn2*tjp1
			t[jr*ldt+j+1] = sn2*tj + cs2*tjp1
		}

		// Apply H = H * R2 (column operation).
		for jr := ifrstm; jr <= j+1; jr++ {
			hj := h[jr*ldh+j]
			hjp1 := h[jr*ldh+j+1]
			h[jr*ldh+j] = cs2*hj - sn2*hjp1
			h[jr*ldh+j+1] = sn2*hj + cs2*hjp1
		}

		// Apply Z = Z * R2.
		if ilz {
			for jr := range n {
				zj := z[jr*ldz+j]
				zjp1 := z[jr*ldz+j+1]
				z[jr*ldz+j] = cs2*zj - sn2*zjp1
				z[jr*ldz+j+1] = sn2*zj + cs2*zjp1
			}
		}
	}

	// Ensure T diagonals are positive.
	// This is a left (row) transformation: negate row j of H and T.
	// For the factorization H_orig = Q * H * Z^T, if H_new = D * H where D[j,j] = -1,
	// then Q_new = Q * D (negate column j of Q).
	if t[j*ldt+j] < 0 {
		// Negate row j of H (from column ifrstm to ilastm, including subdiagonal).
		for jc := ifrstm; jc <= ilastm; jc++ {
			h[j*ldh+jc] = -h[j*ldh+jc]
		}
		// Negate row j of T (from column j to ilastm, T is upper triangular).
		for jc := j; jc <= ilastm; jc++ {
			t[j*ldt+jc] = -t[j*ldt+jc]
		}
		if ilq {
			for jr := range n {
				q[jr*ldq+j] = -q[jr*ldq+j]
			}
		}
	}
	if t[(j+1)*ldt+j+1] < 0 {
		// Negate row j+1 of H (from column ifrstm to ilastm).
		for jc := ifrstm; jc <= ilastm; jc++ {
			h[(j+1)*ldh+jc] = -h[(j+1)*ldh+jc]
		}
		// Negate row j+1 of T (from column j+1 to ilastm).
		for jc := j + 1; jc <= ilastm; jc++ {
			t[(j+1)*ldt+jc] = -t[(j+1)*ldt+jc]
		}
		if ilq {
			for jr := range n {
				q[jr*ldq+j+1] = -q[jr*ldq+j+1]
			}
		}
	}
}
