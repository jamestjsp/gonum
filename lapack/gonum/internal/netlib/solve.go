// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapack
#include <lapack.h>
#include <stdint.h>

_Static_assert(sizeof(lapack_int) == sizeof(int32_t), "Dgetrs bridge requires LP64 lapack_int");

static lapack_int run_dgetrs(char trans, lapack_int n, lapack_int nrhs,
		double *a, lapack_int lda, lapack_int *ipiv, double *b, lapack_int ldb) {
	lapack_int info;
	LAPACK_dgetrs(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
	return info;
}

static lapack_int run_dpotrs(char uplo, lapack_int n, lapack_int nrhs,
		double *a, lapack_int lda, double *b, lapack_int ldb) {
	lapack_int info;
	LAPACK_dpotrs(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info);
	return info;
}
*/
import "C"

import "unsafe"

// Dgetrs solves a system using a column-major LU factorization. Pivots use
// Fortran's one-based convention.
func Dgetrs(trans byte, n, nrhs int, a []float64, lda int, ipiv []int32, b []float64, ldb int) int {
	return int(C.run_dgetrs(C.char(trans), C.lapack_int(n), C.lapack_int(nrhs),
		doublePtr(a), C.lapack_int(lda), (*C.lapack_int)(unsafe.Pointer(unsafe.SliceData(ipiv))),
		doublePtr(b), C.lapack_int(ldb)))
}

// Dpotrs solves a system using a column-major Cholesky factorization.
func Dpotrs(uplo byte, n, nrhs int, a []float64, lda int, b []float64, ldb int) int {
	return int(C.run_dpotrs(C.char(uplo), C.lapack_int(n), C.lapack_int(nrhs),
		doublePtr(a), C.lapack_int(lda), doublePtr(b), C.lapack_int(ldb)))
}
