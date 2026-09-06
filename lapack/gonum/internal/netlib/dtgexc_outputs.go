// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas
#include <lapacke.h>
#include <stdint.h>
#include <stdlib.h>

static lapack_int run_dtgsen_outputs(uint64_t mask, lapack_int n, double *a,
		double *b, double *ar, double *ai, double *beta, double *q, double *z,
		lapack_int *m, double *pl, double *pr, double *dif, double *work0,
		lapack_int *iwork0) {
	lapack_logical *select = malloc((size_t)n*sizeof(lapack_logical));
	for (lapack_int i = 0; i < n; i++) select[i] = (mask >> i) & 1;
	lapack_int lwork = n > 0 ? 4*n + 16 : 16;
	lapack_int liwork = 1;
	double *work = malloc((size_t)lwork*sizeof(double));
	lapack_int *iwork = malloc((size_t)liwork*sizeof(lapack_int));
	lapack_int info = LAPACKE_dtgsen_work(LAPACK_ROW_MAJOR, 0, 1, 1, select,
		n, a, n, b, n, ar, ai, beta, q, n, z, n, m, pl, pr, dif,
		work, lwork, iwork, liwork);
	*work0 = work[0];
	*iwork0 = iwork[0];
	free(select); free(work); free(iwork);
	return info;
}
*/
import "C"

import "unsafe"

func DtgexcOutputs(wantq, wantz bool, n int, a, b, q, z []float64, ifst, ilst int) (ifstOut, ilstOut, info int, work0 float64) {
	cifst, cilst := C.lapack_int(ifst+1), C.lapack_int(ilst+1)
	lwork := 1
	if n > 1 {
		lwork = 4*n + 16
	}
	work := make([]float64, lwork)
	cinfo := C.LAPACKE_dtgexc_work(C.int(C.LAPACK_ROW_MAJOR), C.lapack_logical(boolInt(wantq)),
		C.lapack_logical(boolInt(wantz)), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&a[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&b[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&q[0])), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&z[0])), C.lapack_int(n),
		&cifst, &cilst, (*C.double)(unsafe.Pointer(&work[0])), C.lapack_int(lwork))
	return int(cifst) - 1, int(cilst) - 1, int(cinfo), work[0]
}

func DtgsenReorderOutputs(selected []bool, n int, a, b, ar, ai, beta, q, z []float64) (m, info int, work0 float64, iwork0 int) {
	var mask uint64
	for i, selected := range selected {
		if selected {
			mask |= 1 << i
		}
	}
	var cm, ciwork C.lapack_int
	var cpl, cpr, cwork C.double
	var dif [2]C.double
	cinfo := C.run_dtgsen_outputs(C.uint64_t(mask), C.lapack_int(n),
		(*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		(*C.double)(unsafe.Pointer(&ar[0])), (*C.double)(unsafe.Pointer(&ai[0])),
		(*C.double)(unsafe.Pointer(&beta[0])), (*C.double)(unsafe.Pointer(&q[0])),
		(*C.double)(unsafe.Pointer(&z[0])), &cm, &cpl, &cpr, &dif[0], &cwork, &ciwork)
	return int(cm), int(cinfo), float64(cwork), int(ciwork)
}
