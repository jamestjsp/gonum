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
	ascale := 1 / math.Max(safmin, anorm)
	bscale := 1 / math.Max(safmin, bnorm)

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
	totalMaxit := maxit * (ihi - ilo + 1)
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

		// Handle negligible T diagonals (infinite eigenvalues).
		// Check T(ilast,ilast) first, then scan interior of block.
		if math.Abs(t[ilast*ldt+ilast]) <= btol {
			t[ilast*ldt+ilast] = 0
			goto deflateInfinite
		}

		// Scan for zero T diagonals in the active block [ifirst, ilast-1].
		// Chase any found zero down to T(ilast,ilast) and deflate.
		{
			bi := blas64.Implementation()
			chased := false
			for jj := ilast - 1; jj >= ifirst; jj-- {
				if math.Abs(t[jj*ldt+jj]) > btol {
					continue
				}
				t[jj*ldt+jj] = 0

				if jj == ilo || h[jj*ldh+jj-1] == 0 {
					// H also splits at jj. Chase zero down using
					// left rotations on H (LAPACK DO 40 loop).
					for jch := jj; jch < ilast; jch++ {
						cs, sn, r := impl.Dlartg(h[jch*ldh+jch], h[(jch+1)*ldh+jch])
						h[jch*ldh+jch] = r
						h[(jch+1)*ldh+jch] = 0
						nrot := ilastm - jch
						if nrot > 0 {
							bi.Drot(nrot, h[jch*ldh+jch+1:], 1, h[(jch+1)*ldh+jch+1:], 1, cs, sn)
							bi.Drot(nrot, t[jch*ldt+jch+1:], 1, t[(jch+1)*ldt+jch+1:], 1, cs, sn)
						}
						if ilq {
							bi.Drot(n, q[jch:], ldq, q[jch+1:], ldq, cs, sn)
						}
						if math.Abs(t[(jch+1)*ldt+jch+1]) >= btol {
							// Block split found.
							if jch+1 >= ilast {
								goto deflateInfinite
							}
							ifirst = jch + 1
							chased = true
							break
						}
						t[(jch+1)*ldt+jch+1] = 0
					}
					if chased {
						break
					}
					goto deflateInfinite
				}

				// H does not split at jj. Chase zero from T(jj,jj)
				// down to T(ilast,ilast) using alternating left and
				// right Givens rotations (LAPACK DO 50 loop).
				for jch := jj; jch < ilast; jch++ {
					// Left rotation on rows jch, jch+1 to zero T(jch+1,jch+1).
					cs, sn, r := impl.Dlartg(t[jch*ldt+jch+1], t[(jch+1)*ldt+jch+1])
					t[jch*ldt+jch+1] = r
					t[(jch+1)*ldt+jch+1] = 0
					if jch < ilastm-1 {
						bi.Drot(ilastm-jch-1, t[jch*ldt+jch+2:], 1, t[(jch+1)*ldt+jch+2:], 1, cs, sn)
					}
					bi.Drot(ilastm-jch+2, h[jch*ldh+jch-1:], 1, h[(jch+1)*ldh+jch-1:], 1, cs, sn)
					if ilq {
						bi.Drot(n, q[jch:], ldq, q[jch+1:], ldq, cs, sn)
					}
					// Right rotation to restore H upper Hessenberg.
					cs, sn, r = impl.Dlartg(h[(jch+1)*ldh+jch], h[(jch+1)*ldh+jch-1])
					h[(jch+1)*ldh+jch] = r
					h[(jch+1)*ldh+jch-1] = 0
					bi.Drot(jch+1-ifrstm, h[ifrstm*ldh+jch:], ldh, h[ifrstm*ldh+jch-1:], ldh, cs, sn)
					if jch > ifrstm {
						bi.Drot(jch-ifrstm, t[ifrstm*ldt+jch:], ldt, t[ifrstm*ldt+jch-1:], ldt, cs, sn)
					}
					if ilz {
						bi.Drot(n, z[jch:], ldz, z[jch-1:], ldz, cs, sn)
					}
				}
				goto deflateInfinite
			}
			if chased {
				// DO 40 found a block split; ifirst was updated.
				// Continue to shift computation with the new block.
				goto doQZStep
			}
		}

		goto doQZStep

	deflateInfinite:
		// T(ilast,ilast) = 0: zero H(ilast,ilast-1) via right rotation
		// and deflate as infinite eigenvalue (LAPACK label 70 + 80).
		{
			bi := blas64.Implementation()
			cs, sn, r := impl.Dlartg(h[ilast*ldh+ilast], h[ilast*ldh+ilast-1])
			h[ilast*ldh+ilast] = r
			h[ilast*ldh+ilast-1] = 0
			if ilast-ifrstm > 0 {
				bi.Drot(ilast-ifrstm, h[ifrstm*ldh+ilast:], ldh, h[ifrstm*ldh+ilast-1:], ldh, cs, sn)
				bi.Drot(ilast-ifrstm, t[ifrstm*ldt+ilast:], ldt, t[ifrstm*ldt+ilast-1:], ldt, cs, sn)
			}
			if ilz {
				bi.Drot(n, z[ilast:], ldz, z[ilast-1:], ldz, cs, sn)
			}
			// Standardize sign: T(ilast,ilast) >= 0.
			if t[ilast*ldt+ilast] < 0 {
				if ilschr {
					for j := ifrstm; j <= ilast; j++ {
						h[j*ldh+ilast] = -h[j*ldh+ilast]
						t[j*ldt+ilast] = -t[j*ldt+ilast]
					}
				} else {
					h[ilast*ldh+ilast] = -h[ilast*ldh+ilast]
					t[ilast*ldt+ilast] = -t[ilast*ldt+ilast]
				}
				if ilz {
					for j := range n {
						z[j*ldz+ilast] = -z[j*ldz+ilast]
					}
				}
			}
			alphar[ilast] = h[ilast*ldh+ilast]
			alphai[ilast] = 0
			beta[ilast] = t[ilast*ldt+ilast]
			ilast--
			iiter = 0
			eshift = 0
			continue
		}

	doQZStep:
		// Perform QZ step.
		iiter++

		// Compute shifts from the trailing 2x2 block.
		var s1, s2, wr, wr2, wi float64

		// Exceptional shift every 10 iterations.
		if iiter%10 == 0 {
			if float64(totalMaxit)*safmin*math.Abs(h[ilast*ldh+ilast-1]) <
				math.Abs(t[(ilast-1)*ldt+ilast-1]) {
				eshift = h[ilast*ldh+ilast-1] / t[(ilast-1)*ldt+ilast-1]
			} else {
				eshift += 1 / (safmin * float64(totalMaxit))
			}
			s1 = 1
			wr = eshift
			wi = 0
		} else {
			// Normal shift: eigenvalues of trailing 2x2.
			s1, s2, wr, wr2, wi = impl.Dlag2(h[(ilast-1)*ldh+ilast-1:], ldh, t[(ilast-1)*ldt+ilast-1:], ldt)
			_, _ = wr2, s2
		}

		// Do one QZ sweep.
		if wi == 0 {
			impl.doQZSweepSingle(ilschr, ilq, ilz, n, ifirst, ilast, ifrstm, ilastm,
				h, ldh, t, ldt, q, ldq, z, ldz, s1, wr, safmin)
		} else {
			// Double-shift requires nonzero T diagonals for scaled pencil formula.
			// Fall back to single-shift for singular pencils.
			hasSmallTDiag := false
			for k := ifirst; k <= ilast; k++ {
				if math.Abs(t[k*ldt+k]) <= btol {
					hasSmallTDiag = true
					break
				}
			}
			if hasSmallTDiag {
				impl.doQZSweepSingle(ilschr, ilq, ilz, n, ifirst, ilast, ifrstm, ilastm,
					h, ldh, t, ldt, q, ldq, z, ldz, s1, wr, safmin)
			} else {
				impl.doQZSweepDouble(ilschr, ilq, ilz, n, ifirst, ilast, ifrstm, ilastm,
					h, ldh, t, ldt, q, ldq, z, ldz, ascale, bscale, safmin)
			}
		}
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

	bi := blas64.Implementation()
	istart := ifirst

	temp := h[istart*ldh+istart]
	if s1 != 0 {
		temp -= (wr / s1) * t[istart*ldt+istart]
	}
	temp2 := h[(istart+1)*ldh+istart]

	cs, sn, _ := impl.Dlartg(temp, temp2)

	for j := istart; j < ilast; j++ {
		if j > istart {
			temp = h[j*ldh+j-1]
			temp2 = h[(j+1)*ldh+j-1]
			cs, sn, _ = impl.Dlartg(temp, temp2)
			h[j*ldh+j-1] = cs*temp + sn*temp2
			h[(j+1)*ldh+j-1] = 0
		}

		nj := ilastm - j + 1
		bi.Drot(nj, h[j*ldh+j:], 1, h[(j+1)*ldh+j:], 1, cs, sn)
		bi.Drot(nj, t[j*ldt+j:], 1, t[(j+1)*ldt+j:], 1, cs, sn)
		if ilq {
			bi.Drot(n, q[j:], ldq, q[j+1:], ldq, cs, sn)
		}

		if t[(j+1)*ldt+j] != 0 {
			cs, sn, _ = impl.Dlartg(t[(j+1)*ldt+j+1], t[(j+1)*ldt+j])
			t[(j+1)*ldt+j+1] = cs*t[(j+1)*ldt+j+1] + sn*t[(j+1)*ldt+j]
			t[(j+1)*ldt+j] = 0

			nh := min(j+2, ilast) - ifrstm + 1
			bi.Drot(nh, h[ifrstm*ldh+j+1:], ldh, h[ifrstm*ldh+j:], ldh, cs, sn)
			nt := j - ifrstm + 1
			if nt > 0 {
				bi.Drot(nt, t[ifrstm*ldt+j+1:], ldt, t[ifrstm*ldt+j:], ldt, cs, sn)
			}
			if ilz {
				bi.Drot(n, z[j+1:], ldz, z[j:], ldz, cs, sn)
			}
		}
	}
}

// doQZSweepDouble performs an implicit double-shift QZ sweep for complex
// conjugate shifts. It follows the Francis implicit double-shift QZ algorithm
// as described in LAPACK's dhgeqz.f.
func (impl Implementation) doQZSweepDouble(ilschr, ilq, ilz bool, n, ifirst, ilast, ifrstm, ilastm int,
	h []float64, ldh int, t []float64, ldt int, q []float64, ldq int, z []float64, ldz int,
	ascale, bscale, safmin float64) {

	istart := ifirst

	// Compute first column of double-shift polynomial using LAPACK's
	// numerically stable formula based on scaled pencil entries.
	// Bottom-right 2x2 of T^{-1}*H (scaled).
	ad11 := (ascale * h[(ilast-1)*ldh+ilast-1]) / (bscale * t[(ilast-1)*ldt+ilast-1])
	ad21 := (ascale * h[ilast*ldh+ilast-1]) / (bscale * t[(ilast-1)*ldt+ilast-1])
	ad12 := (ascale * h[(ilast-1)*ldh+ilast]) / (bscale * t[ilast*ldt+ilast])
	ad22 := (ascale * h[ilast*ldh+ilast]) / (bscale * t[ilast*ldt+ilast])
	u12 := t[(ilast-1)*ldt+ilast] / t[ilast*ldt+ilast]

	// Top-left 3x3 of T^{-1}*H (scaled).
	ad11l := (ascale * h[istart*ldh+istart]) / (bscale * t[istart*ldt+istart])
	ad21l := (ascale * h[(istart+1)*ldh+istart]) / (bscale * t[istart*ldt+istart])
	ad12l := (ascale * h[istart*ldh+istart+1]) / (bscale * t[(istart+1)*ldt+istart+1])
	ad22l := (ascale * h[(istart+1)*ldh+istart+1]) / (bscale * t[(istart+1)*ldt+istart+1])
	ad32l := (ascale * h[(istart+2)*ldh+istart+1]) / (bscale * t[(istart+1)*ldt+istart+1])
	u12l := t[istart*ldt+istart+1] / t[(istart+1)*ldt+istart+1]

	v1 := (ad11-ad11l)*(ad22-ad11l) - ad12*ad21 +
		ad21*u12*ad11l + (ad12l-ad11l*u12l)*ad21l
	v2 := ((ad22l - ad11l) - ad21l*u12l - (ad11 - ad11l) -
		(ad22 - ad11l) + ad21*u12) * ad21l
	v3 := ad32l * ad21l

	// Create Householder transformation to introduce bulge.
	var vv [2]float64
	vv[0] = v2
	vv[1] = v3
	v1, tau := impl.Dlarfg(3, v1, vv[:], 1)
	_ = v1
	var v [3]float64
	v[0] = 1
	v[1] = vv[0]
	v[2] = vv[1]

	// Chase the bulge through the matrix.
	for j := istart; j < ilast-1; j++ {
		if j > istart {
			v1 = h[j*ldh+j-1]
			v2 = h[(j+1)*ldh+j-1]
			v3 = h[(j+2)*ldh+j-1]
			vv[0] = v2
			vv[1] = v3
			v1, tau = impl.Dlarfg(3, v1, vv[:], 1)
			v[0] = 1
			v[1] = vv[0]
			v[2] = vv[1]

			h[j*ldh+j-1] = v1
			h[(j+1)*ldh+j-1] = 0
			h[(j+2)*ldh+j-1] = 0
		}

		// Apply Householder from left to H and T.
		t2 := tau * v[1]
		t3 := tau * v[2]
		for jc := j; jc <= ilastm; jc++ {
			sumh := h[j*ldh+jc] + v[1]*h[(j+1)*ldh+jc] + v[2]*h[(j+2)*ldh+jc]
			h[j*ldh+jc] -= tau * sumh
			h[(j+1)*ldh+jc] -= t2 * sumh
			h[(j+2)*ldh+jc] -= t3 * sumh

			sumt := t[j*ldt+jc] + v[1]*t[(j+1)*ldt+jc] + v[2]*t[(j+2)*ldt+jc]
			t[j*ldt+jc] -= tau * sumt
			t[(j+1)*ldt+jc] -= t2 * sumt
			t[(j+2)*ldt+jc] -= t3 * sumt
		}
		if ilq {
			for jr := 0; jr < n; jr++ {
				sum := q[jr*ldq+j] + v[1]*q[jr*ldq+j+1] + v[2]*q[jr*ldq+j+2]
				q[jr*ldq+j] -= tau * sum
				q[jr*ldq+j+1] -= t2 * sum
				q[jr*ldq+j+2] -= t3 * sum
			}
		}

		// Restore T to upper triangular form.
		// Solve 2x2 system to find Householder that zeros T[j+1,j] and T[j+2,j]
		// simultaneously (LAPACK DLAGBC approach).
		ilpivt := false
		tmp1 := math.Max(math.Abs(t[(j+1)*ldt+j+1]), math.Abs(t[(j+1)*ldt+j+2]))
		tmp2 := math.Max(math.Abs(t[(j+2)*ldt+j+1]), math.Abs(t[(j+2)*ldt+j+2]))

		var w11, w12, w21, w22, u1, u2, scl float64
		if math.Max(tmp1, tmp2) < safmin {
			scl = 0
			u1 = 1
			u2 = 0
		} else {
			if tmp1 >= tmp2 {
				w11 = t[(j+1)*ldt+j+1]
				w21 = t[(j+2)*ldt+j+1]
				w12 = t[(j+1)*ldt+j+2]
				w22 = t[(j+2)*ldt+j+2]
				u1 = t[(j+1)*ldt+j]
				u2 = t[(j+2)*ldt+j]
			} else {
				w21 = t[(j+1)*ldt+j+1]
				w11 = t[(j+2)*ldt+j+1]
				w22 = t[(j+1)*ldt+j+2]
				w12 = t[(j+2)*ldt+j+2]
				u2 = t[(j+1)*ldt+j]
				u1 = t[(j+2)*ldt+j]
			}

			if math.Abs(w12) > math.Abs(w11) {
				ilpivt = true
				w11, w12 = w12, w11
				w21, w22 = w22, w21
			}

			tmp := w21 / w11
			u2 -= tmp * u1
			w22 -= tmp * w12

			scl = 1
			if math.Abs(w22) < safmin {
				scl = 0
				u2 = 1
				u1 = -w12 / w11
			} else {
				if math.Abs(w22) < math.Abs(u2) {
					scl = math.Abs(w22 / u2)
				}
				if math.Abs(w11) < math.Abs(u1) {
					scl = math.Min(scl, math.Abs(w11/u1))
				}
				u2 = (scl * u2) / w22
				u1 = (scl*u1 - w12*u2) / w11
			}
		}

		if ilpivt {
			u1, u2 = u2, u1
		}

		// Compute right Householder vector from [scl, u1, u2].
		t1 := math.Sqrt(scl*scl + u1*u1 + u2*u2)
		tauR := 1 + scl/t1
		vs := -1 / (scl + t1)
		v[0] = 1
		v[1] = vs * u1
		v[2] = vs * u2

		// Apply right Householder to H, T, Z.
		t2 = tauR * v[1]
		t3 = tauR * v[2]
		for jr := ifrstm; jr <= j+2; jr++ {
			sumh := h[jr*ldh+j] + v[1]*h[jr*ldh+j+1] + v[2]*h[jr*ldh+j+2]
			h[jr*ldh+j] -= tauR * sumh
			h[jr*ldh+j+1] -= t2 * sumh
			h[jr*ldh+j+2] -= t3 * sumh

			sumt := t[jr*ldt+j] + v[1]*t[jr*ldt+j+1] + v[2]*t[jr*ldt+j+2]
			t[jr*ldt+j] -= tauR * sumt
			t[jr*ldt+j+1] -= t2 * sumt
			t[jr*ldt+j+2] -= t3 * sumt
		}
		if j+3 <= ilast {
			jr := j + 3
			sumh := h[jr*ldh+j] + v[1]*h[jr*ldh+j+1] + v[2]*h[jr*ldh+j+2]
			h[jr*ldh+j] -= tauR * sumh
			h[jr*ldh+j+1] -= t2 * sumh
			h[jr*ldh+j+2] -= t3 * sumh
		}
		if ilz {
			for jr := 0; jr < n; jr++ {
				sum := z[jr*ldz+j] + v[1]*z[jr*ldz+j+1] + v[2]*z[jr*ldz+j+2]
				z[jr*ldz+j] -= tauR * sum
				z[jr*ldz+j+1] -= t2 * sum
				z[jr*ldz+j+2] -= t3 * sum
			}
		}
		t[(j+1)*ldt+j] = 0
		t[(j+2)*ldt+j] = 0
	}

	// Final step: handle the last 2x2 portion with Givens rotations.
	bi := blas64.Implementation()
	j := ilast - 1
	tmp := h[j*ldh+j-1]
	tmp2 := h[(j+1)*ldh+j-1]
	cs, sn, r := impl.Dlartg(tmp, tmp2)
	h[j*ldh+j-1] = r
	h[(j+1)*ldh+j-1] = 0

	nj := ilastm - j + 1
	bi.Drot(nj, h[j*ldh+j:], 1, h[(j+1)*ldh+j:], 1, cs, sn)
	bi.Drot(nj, t[j*ldt+j:], 1, t[(j+1)*ldt+j:], 1, cs, sn)
	if ilq {
		bi.Drot(n, q[j:], ldq, q[j+1:], ldq, cs, sn)
	}

	// Restore T triangular for the final 2x2 step.
	tmp = t[(j+1)*ldt+j+1]
	cs, sn, r = impl.Dlartg(tmp, t[(j+1)*ldt+j])
	t[(j+1)*ldt+j+1] = r
	t[(j+1)*ldt+j] = 0

	nh := ilast - ifrstm + 1
	bi.Drot(nh, h[ifrstm*ldh+j+1:], ldh, h[ifrstm*ldh+j:], ldh, cs, sn)
	nt := ilast - 1 - ifrstm + 1
	if nt > 0 {
		bi.Drot(nt, t[ifrstm*ldt+j+1:], ldt, t[ifrstm*ldt+j:], ldt, cs, sn)
	}
	if ilz {
		bi.Drot(n, z[j+1:], ldz, z[j:], ldz, cs, sn)
	}
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

	bi := blas64.Implementation()

	// Apply Dlanv2 transformation to rest of H from left: rows j, j+1.
	nh := ilastm - (j + 2) + 1
	if nh > 0 {
		bi.Drot(nh, h[j*ldh+j+2:], 1, h[(j+1)*ldh+j+2:], 1, cs, sn)
	}
	// Apply from right: columns j, j+1.
	nh = j - ifrstm
	if nh > 0 {
		bi.Drot(nh, h[ifrstm*ldh+j:], ldh, h[ifrstm*ldh+j+1:], ldh, cs, sn)
	}
	// Apply to T from left: rows j, j+1.
	nh = ilastm - j + 1
	bi.Drot(nh, t[j*ldt+j:], 1, t[(j+1)*ldt+j:], 1, cs, sn)
	// Apply to T from right: columns j, j+1.
	nh = j + 2 - ifrstm
	if nh > 0 {
		bi.Drot(nh, t[ifrstm*ldt+j:], ldt, t[ifrstm*ldt+j+1:], ldt, cs, sn)
	}
	if ilq {
		bi.Drot(n, q[j:], ldq, q[j+1:], ldq, cs, sn)
	}
	if ilz {
		bi.Drot(n, z[j:], ldz, z[j+1:], ldz, cs, sn)
	}

	// Make T upper triangular with positive diagonal.
	// After applying the Dlanv2 similarity transformation, T may have
	// nonzero t21. We need to eliminate it using a right rotation.
	// Also ensure T diagonals are positive.

	t21 := t[(j+1)*ldt+j]

	// Eliminate t21 using a right rotation on columns j, j+1.
	// Find R2 = [cs2 sn2; -sn2 cs2] such that [t21, t22] * R2 = [0, *]
	// This means cs2*t21 - sn2*t22 = 0.
	if t21 != 0 {
		cs2, sn2, _ := impl.Dlartg(t[(j+1)*ldt+j+1], t21)

		// Apply T = T * R2 (column operation).
		bi.Drot(j+2-ifrstm, t[ifrstm*ldt+j:], ldt, t[ifrstm*ldt+j+1:], ldt, cs2, -sn2)
		t[(j+1)*ldt+j] = 0

		// Apply H = H * R2 (column operation).
		bi.Drot(j+2-ifrstm, h[ifrstm*ldh+j:], ldh, h[ifrstm*ldh+j+1:], ldh, cs2, -sn2)

		// Apply Z = Z * R2.
		if ilz {
			bi.Drot(n, z[j:], ldz, z[j+1:], ldz, cs2, -sn2)
		}
	}

	// Ensure T diagonals are positive.
	// This is a left (row) transformation: negate row j of H and T.
	// For the factorization H_orig = Q * H * Z^T, if H_new = D * H where D[j,j] = -1,
	// then Q_new = Q * D (negate column j of Q).
	if t[j*ldt+j] < 0 {
		bi.Dscal(ilastm-ifrstm+1, -1, h[j*ldh+ifrstm:], 1)
		bi.Dscal(ilastm-j+1, -1, t[j*ldt+j:], 1)
		if ilq {
			bi.Dscal(n, -1, q[j:], ldq)
		}
	}
	if t[(j+1)*ldt+j+1] < 0 {
		bi.Dscal(ilastm-ifrstm+1, -1, h[(j+1)*ldh+ifrstm:], 1)
		bi.Dscal(ilastm-j, -1, t[(j+1)*ldt+j+1:], 1)
		if ilq {
			bi.Dscal(n, -1, q[j+1:], ldq)
		}
	}
}
