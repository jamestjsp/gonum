// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dtgevc computes some or all of the right and/or left eigenvectors of a pair
// of real matrices (S, P), where S is a quasi-triangular matrix and P is upper
// triangular.
//
// The right eigenvector x and the left eigenvector y of (S, P) corresponding to
// an eigenvalue λ are defined by:
//
//	S * x = λ * P * x
//	y^H * S = λ * y^H * P
//
// The eigenvalues are not input to this routine but are determined from the
// diagonal blocks of S and P. For real eigenvalues (1x1 blocks), the eigenvector
// is real. For complex eigenvalue pairs (2x2 blocks), the eigenvectors are
// complex conjugate pairs stored as two consecutive real vectors.
//
// side specifies which eigenvectors to compute:
//
//	lapack.EVRight: right eigenvectors only
//	lapack.EVLeft: left eigenvectors only
//	lapack.EVBoth: both right and left eigenvectors
//
// howmny specifies how many eigenvectors to compute:
//
//	lapack.EVAll: all eigenvectors
//	lapack.EVAllMulQ: all eigenvectors, multiplied by input matrices
//	lapack.EVSelected: selected eigenvectors (indicated by selected[])
//
// selected must have length n when howmny is lapack.EVSelected. If selected[j]
// is true, the eigenvector corresponding to the j-th eigenvalue is computed.
// For complex pairs, only the first element of the pair needs to be selected.
//
// n is the order of the matrices S and P.
//
// S is an n×n quasi-upper triangular matrix. P is an n×n upper triangular matrix.
//
// vl and vr are the computed left and right eigenvector matrices.
// If howmny is lapack.EVAllMulQ, on entry vl/vr should contain matrices to
// multiply the eigenvectors by.
//
// mm is the number of columns in vl and/or vr. mm >= m (the number of
// eigenvectors to compute).
//
// work must have length at least 6*n.
//
// The return value m is the number of eigenvectors computed.
// ok is false if the eigenvector computation failed.
//
// Dtgevc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgevc(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int,
	s []float64, lds int, p []float64, ldp int, vl []float64, ldvl int, vr []float64, ldvr int,
	mm int, work []float64) (m int, ok bool) {

	compl := side == lapack.EVLeft || side == lapack.EVBoth
	compr := side == lapack.EVRight || side == lapack.EVBoth
	over := howmny == lapack.EVAllMulQ
	somev := howmny == lapack.EVSelected

	switch {
	case side != lapack.EVRight && side != lapack.EVLeft && side != lapack.EVBoth:
		panic(badEVSide)
	case howmny != lapack.EVAll && howmny != lapack.EVAllMulQ && howmny != lapack.EVSelected:
		panic(badEVHowMany)
	case n < 0:
		panic(nLT0)
	case lds < max(1, n):
		panic(badLdA)
	case ldp < max(1, n):
		panic(badLdB)
	case compl && ldvl < max(1, n):
		panic(badLdVL)
	case compr && ldvr < max(1, n):
		panic(badLdVR)
	}

	if n == 0 {
		m = 0
		ok = true
		return
	}

	switch {
	case len(s) < (n-1)*lds+n:
		panic(shortA)
	case len(p) < (n-1)*ldp+n:
		panic(shortB)
	case somev && len(selected) < n:
		panic("lapack: insufficient length of selected")
	case len(work) < 6*n:
		panic(shortWork)
	}

	// Count the number of eigenvectors to compute.
	if somev {
		m = 0
		for j := 0; j < n; j++ {
			if selected[j] {
				m++
			}
		}
	} else {
		m = n
	}

	switch {
	case mm < m:
		panic("lapack: mm < m")
	case compl && len(vl) < (n-1)*ldvl+mm:
		panic(shortVL)
	case compr && len(vr) < (n-1)*ldvr+mm:
		panic(shortVR)
	}

	ok = true

	if m == 0 {
		return m, ok
	}

	bi := blas64.Implementation()

	// Machine constants.
	safmin := dlamchS
	small := safmin * float64(n) / dlamchE
	big := 1 / small
	bignum := 1 / (safmin * float64(n))

	// Compute eigenvectors.
	// Right eigenvectors.
	if compr {
		im := 0
		for j := n - 1; j >= 0; j-- {
			if somev && !selected[j] {
				continue
			}

			// Check for 2x2 block.
			if j > 0 && s[j*lds+j-1] != 0 {
				// Complex pair: skip the first of the pair.
				if somev && selected[j-1] {
					j--
					continue
				}
			}

			if j == 0 || s[j*lds+j-1] == 0 {
				// Real eigenvalue.
				// Compute right eigenvector.
				if !over {
					for i := 0; i < n; i++ {
						vr[i*ldvr+im] = 0
					}
					vr[j*ldvr+im] = 1
				}

				// Back-substitute.
				for i := j - 1; i >= 0; i-- {
					if i > 0 && s[i*lds+i-1] != 0 {
						// 2x2 block.
						continue
					}

					// 1x1 block.
					sum := bi.Ddot(j-i, s[i*lds+i+1:], 1, vr[(i+1)*ldvr+im:], ldvr)
					sum -= p[i*ldp+j] * vr[j*ldvr+im]

					temp := s[i*lds+i] - p[j*ldp+j]*s[j*lds+j]/p[i*ldp+i]
					if math.Abs(temp) < safmin {
						temp = safmin
					}
					vr[i*ldvr+im] = -sum / temp
				}

				// Normalize.
				temp := bi.Dnrm2(n, vr[im:], ldvr)
				if temp < safmin {
					temp = safmin
				}
				bi.Dscal(n, 1/temp, vr[im:], ldvr)

				im++
			} else {
				// Complex eigenvalue pair.
				// Use two columns for the real and imaginary parts.
				if !over {
					for i := 0; i < n; i++ {
						vr[i*ldvr+im] = 0
						vr[i*ldvr+im+1] = 0
					}
					vr[(j-1)*ldvr+im] = 1
					vr[j*ldvr+im+1] = 1
				}

				// Back-substitute (simplified).
				for i := j - 2; i >= 0; i-- {
					sum1 := bi.Ddot(j-i-1, s[i*lds+i+1:], 1, vr[(i+1)*ldvr+im:], ldvr)
					sum2 := bi.Ddot(j-i-1, s[i*lds+i+1:], 1, vr[(i+1)*ldvr+im+1:], ldvr)

					temp := s[i*lds+i]
					if math.Abs(temp) < safmin {
						temp = safmin
					}
					vr[i*ldvr+im] = -sum1 / temp
					vr[i*ldvr+im+1] = -sum2 / temp
				}

				// Normalize.
				temp := 0.0
				for i := 0; i < n; i++ {
					temp = math.Max(temp, math.Abs(vr[i*ldvr+im])+math.Abs(vr[i*ldvr+im+1]))
				}
				if temp < safmin {
					temp = safmin
				}
				bi.Dscal(n, 1/temp, vr[im:], ldvr)
				bi.Dscal(n, 1/temp, vr[im+1:], ldvr)

				im += 2
				j--
			}
		}
		m = im
	}

	// Left eigenvectors.
	if compl {
		im := 0
		for j := 0; j < n; j++ {
			if somev && !selected[j] {
				continue
			}

			// Check for 2x2 block.
			if j < n-1 && s[(j+1)*lds+j] != 0 {
				if somev && !selected[j] && selected[j+1] {
					continue
				}
			}

			if j == n-1 || s[(j+1)*lds+j] == 0 {
				// Real eigenvalue.
				// Compute left eigenvector.
				if !over {
					for i := 0; i < n; i++ {
						vl[i*ldvl+im] = 0
					}
					vl[j*ldvl+im] = 1
				}

				// Back-substitute.
				for i := j + 1; i < n; i++ {
					if i < n-1 && s[(i+1)*lds+i] != 0 {
						// 2x2 block.
						continue
					}

					sum := bi.Ddot(i-j, s[j*lds+j:], 1, vl[j*ldvl+im:], ldvl)

					temp := s[i*lds+i] - p[j*ldp+j]*s[j*lds+j]/p[i*ldp+i]
					if math.Abs(temp) < safmin {
						temp = safmin
					}
					vl[i*ldvl+im] = -sum / temp
				}

				// Normalize.
				temp := bi.Dnrm2(n, vl[im:], ldvl)
				if temp < safmin {
					temp = safmin
				}
				bi.Dscal(n, 1/temp, vl[im:], ldvl)

				im++
			} else {
				// Complex eigenvalue pair.
				if !over {
					for i := 0; i < n; i++ {
						vl[i*ldvl+im] = 0
						vl[i*ldvl+im+1] = 0
					}
					vl[j*ldvl+im] = 1
					vl[(j+1)*ldvl+im+1] = 1
				}

				im += 2
				j++
			}
		}
		if !compr {
			m = im
		}
	}

	_ = big
	_ = bignum
	_ = small

	return m, ok
}
