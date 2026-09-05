// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

// Package netlib provides an optional differential-test oracle. Source review is
// pinned to Reference-LAPACK v3.12.1 commit 6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca;
// Version reports the independently installed runtime version.
package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas
#include <lapacke.h>

extern void dlasr_(char*, char*, char*, lapack_int*, lapack_int*, double*,
	double*, double*, lapack_int*
#ifdef LAPACK_FORTRAN_STRLEN_END
	, FORTRAN_STRLEN, FORTRAN_STRLEN, FORTRAN_STRLEN
#endif
);

static void run_ilaver(lapack_int *major, lapack_int *minor, lapack_int *patch) {
	LAPACK_ilaver(major, minor, patch);
}

static void run_dlasr(char side, char pivot, char direct, lapack_int m,
		lapack_int n, double *c, double *s, double *a, lapack_int lda) {
	dlasr_(&side, &pivot, &direct, &m, &n, c, s, a, &lda
#ifdef LAPACK_FORTRAN_STRLEN_END
		, 1, 1, 1
#endif
	);
}

static lapack_int run_dbdsqr(char uplo, lapack_int n, lapack_int ncvt,
		lapack_int nru, lapack_int ncc, double *d, double *e, double *vt,
		lapack_int ldvt, double *u, lapack_int ldu, double *c, lapack_int ldc,
		double *work) {
	lapack_int info;
	LAPACK_dbdsqr(&uplo, &n, &ncvt, &nru, &ncc, d, e, vt, &ldvt, u, &ldu,
		c, &ldc, work, &info);
	return info;
}

static lapack_int run_dgesvd(char jobu, char jobvt, lapack_int m, lapack_int n,
		double *a, lapack_int lda, double *s, double *u, lapack_int ldu,
		double *vt, lapack_int ldvt, double *work, lapack_int lwork) {
	lapack_int info;
	LAPACK_dgesvd(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt,
		work, &lwork, &info);
	return info;
}
*/
import "C"

import "unsafe"

func Version() (major, minor, patch int) {
	var cmajor, cminor, cpatch C.lapack_int
	C.run_ilaver(&cmajor, &cminor, &cpatch)
	return int(cmajor), int(cminor), int(cpatch)
}

// Dlasr applies rotations to a column-major matrix without layout conversion.
func Dlasr(side, pivot, direct byte, m, n int, c, s, a []float64, lda int) {
	C.run_dlasr(C.char(side), C.char(pivot), C.char(direct), C.lapack_int(m),
		C.lapack_int(n), doublePtr(c), doublePtr(s), doublePtr(a), C.lapack_int(lda))
}

// Dbdsqr computes the SVD of a bidiagonal matrix. Matrix operands are column-major.
func Dbdsqr(uplo byte, n, ncvt, nru, ncc int, d, e, vt []float64, ldvt int,
	u []float64, ldu int, c []float64, ldc int, work []float64) int {
	return int(C.run_dbdsqr(C.char(uplo), C.lapack_int(n), C.lapack_int(ncvt),
		C.lapack_int(nru), C.lapack_int(ncc), doublePtr(d), doublePtr(e),
		doublePtr(vt), C.lapack_int(ldvt), doublePtr(u), C.lapack_int(ldu),
		doublePtr(c), C.lapack_int(ldc), doublePtr(work)))
}

// Dgesvd computes an SVD in-place. Matrix operands are column-major, so callers
// can keep layout conversion outside kernel benchmarks.
func Dgesvd(jobU, jobVT byte, m, n int, a []float64, lda int, s, u []float64,
	ldu int, vt []float64, ldvt int, work []float64, lwork int) int {
	return int(C.run_dgesvd(C.char(jobU), C.char(jobVT), C.lapack_int(m),
		C.lapack_int(n), doublePtr(a), C.lapack_int(lda), doublePtr(s),
		doublePtr(u), C.lapack_int(ldu), doublePtr(vt), C.lapack_int(ldvt),
		doublePtr(work), C.lapack_int(lwork)))
}

func doublePtr(x []float64) *C.double {
	if len(x) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&x[0]))
}
