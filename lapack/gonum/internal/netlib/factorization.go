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

_Static_assert(sizeof(lapack_int) == sizeof(int32_t), "Dgetrf bridge requires LP64 lapack_int");

static lapack_int run_dgeqrf(lapack_int m, lapack_int n, double *a,
		lapack_int lda, double *tau, double *work, lapack_int lwork) {
	lapack_int info;
	LAPACK_dgeqrf(&m, &n, a, &lda, tau, work, &lwork, &info);
	return info;
}

static lapack_int run_dgetrf(lapack_int m, lapack_int n, double *a,
		lapack_int lda, lapack_int *ipiv) {
	lapack_int info;
	LAPACK_dgetrf(&m, &n, a, &lda, ipiv, &info);
	return info;
}

static lapack_int run_dpotrf(char uplo, lapack_int n, double *a, lapack_int lda) {
	lapack_int info;
	LAPACK_dpotrf(&uplo, &n, a, &lda, &info);
	return info;
}
*/
import "C"

import "unsafe"

// Dgeqrf computes a QR factorization of a column-major matrix.
func Dgeqrf(m, n int, a []float64, lda int, tau, work []float64, lwork int) int {
	return int(C.run_dgeqrf(C.lapack_int(m), C.lapack_int(n), doublePtr(a),
		C.lapack_int(lda), doublePtr(tau), doublePtr(work), C.lapack_int(lwork)))
}

// Dgetrf computes an LU factorization of a column-major matrix. The returned
// pivots retain Fortran's one-based convention to avoid benchmark conversions.
func Dgetrf(m, n int, a []float64, lda int, ipiv []int32) int {
	return int(C.run_dgetrf(C.lapack_int(m), C.lapack_int(n), doublePtr(a),
		C.lapack_int(lda), (*C.lapack_int)(unsafe.Pointer(unsafe.SliceData(ipiv)))))
}

// Dpotrf computes a Cholesky factorization of a column-major matrix.
func Dpotrf(uplo byte, n int, a []float64, lda int) int {
	return int(C.run_dpotrf(C.char(uplo), C.lapack_int(n), doublePtr(a), C.lapack_int(lda)))
}
