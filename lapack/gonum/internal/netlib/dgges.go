// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapack
#include <lapack.h>
#include <math.h>
#include <stdint.h>

_Static_assert(sizeof(lapack_int) == sizeof(int32_t), "Dgges bridge requires LP64 lapack_int");
_Static_assert(sizeof(lapack_logical) == sizeof(int32_t), "Dgges bridge requires 32-bit lapack_logical");

static lapack_logical select_negative_real(const double *ar, const double *ai, const double *beta) {
	(void)ai;
	return (*ar < 0 && *beta > 0) || (*ar > 0 && *beta < 0);
}

static lapack_logical select_unit_disk(const double *ar, const double *ai, const double *beta) {
	return hypot(*ar, *ai) < fabs(*beta);
}

typedef struct {
	lapack_int sdim;
	lapack_int info;
} dgges_result;

static dgges_result run_dgges_work(char jobvsl, char jobvsr, char selection,
		lapack_int n, double *a, double *b, double *ar,
		double *ai, double *beta, double *vsl, double *vsr, double *work,
		lapack_int lwork, lapack_logical *bwork) {
	char sort = 'N';
	LAPACK_D_SELECT3 select = NULL;
	if (selection == 'L') {
		sort = 'S';
		select = select_negative_real;
	} else if (selection == 'D') {
		sort = 'S';
		select = select_unit_disk;
	}
	lapack_int ld = n > 0 ? n : 1;
	dgges_result result = {0, 0};
	LAPACK_dgges(&jobvsl, &jobvsr, &sort, select, &n, a, &ld, b, &ld,
		&result.sdim, ar, ai, beta, vsl, &ld, vsr, &ld, work, &lwork, bwork,
		&result.info);
	return result;
}
*/
import "C"

import "unsafe"

// DggesWork computes a generalized Schur factorization in-place. Matrix
// operands are column-major with leading dimension max(1,n).
func DggesWork(jobvsl, jobvsr, selection byte, n int, a, b, ar, ai, beta, vsl, vsr, work []float64, lwork int, bwork []int32) (sdim, info int) {
	if selection != 'N' && selection != 'L' && selection != 'D' {
		panic("netlib: invalid Dgges selection")
	}
	result := C.run_dgges_work(C.char(jobvsl), C.char(jobvsr), C.char(selection),
		C.lapack_int(n), doublePtr(a), doublePtr(b), doublePtr(ar),
		doublePtr(ai), doublePtr(beta), doublePtr(vsl), doublePtr(vsr),
		doublePtr(work), C.lapack_int(lwork),
		(*C.lapack_logical)(unsafe.Pointer(unsafe.SliceData(bwork))))
	return int(result.sdim), int(result.info)
}
