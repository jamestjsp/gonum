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

_Static_assert(sizeof(lapack_int) == sizeof(int32_t), "QR bridge requires LP64 lapack_int");

static void run_dlarft(char direct, char storev, lapack_int n, lapack_int k,
		double *v, lapack_int ldv, double *tau, double *t, lapack_int ldt) {
	LAPACK_dlarft(&direct, &storev, &n, &k, v, &ldv, tau, t, &ldt);
}

static void run_dlarfb(char side, char trans, char direct, char storev,
		lapack_int m, lapack_int n, lapack_int k, double *v, lapack_int ldv,
		double *t, lapack_int ldt, double *c, lapack_int ldc,
		double *work, lapack_int ldwork) {
	LAPACK_dlarfb(&side, &trans, &direct, &storev, &m, &n, &k, v, &ldv,
		t, &ldt, c, &ldc, work, &ldwork);
}

static lapack_int run_dormqr(char side, char trans, lapack_int m,
		lapack_int n, lapack_int k, double *a, lapack_int lda, double *tau,
		double *c, lapack_int ldc, double *work, lapack_int lwork) {
	lapack_int info;
	LAPACK_dormqr(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc,
		work, &lwork, &info);
	return info;
}
*/
import "C"

// Dlarft forms the triangular factor of a block reflector stored in column-major order.
func Dlarft(direct, store byte, n, k int, v []float64, ldv int, tau, t []float64, ldt int) {
	C.run_dlarft(C.char(direct), C.char(store), C.lapack_int(n), C.lapack_int(k),
		doublePtr(v), C.lapack_int(ldv), doublePtr(tau), doublePtr(t), C.lapack_int(ldt))
}

// Dlarfb applies a block reflector to a column-major matrix.
func Dlarfb(side, trans, direct, store byte, m, n, k int, v []float64, ldv int,
	t []float64, ldt int, c []float64, ldc int, work []float64, ldwork int,
) {
	C.run_dlarfb(C.char(side), C.char(trans), C.char(direct), C.char(store),
		C.lapack_int(m), C.lapack_int(n), C.lapack_int(k), doublePtr(v),
		C.lapack_int(ldv), doublePtr(t), C.lapack_int(ldt), doublePtr(c),
		C.lapack_int(ldc), doublePtr(work), C.lapack_int(ldwork))
}

// Dormqr applies the orthogonal factor of a QR factorization to a column-major matrix.
func Dormqr(side, trans byte, m, n, k int, a []float64, lda int, tau,
	c []float64, ldc int, work []float64, lwork int,
) int {
	return int(C.run_dormqr(C.char(side), C.char(trans), C.lapack_int(m),
		C.lapack_int(n), C.lapack_int(k), doublePtr(a), C.lapack_int(lda),
		doublePtr(tau), doublePtr(c), C.lapack_int(ldc), doublePtr(work),
		C.lapack_int(lwork)))
}
