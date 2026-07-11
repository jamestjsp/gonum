// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas
#include <lapacke.h>
#include <stdlib.h>

static lapack_logical select_negative(const double *ar, const double *ai, const double *beta) {
	(void)ai;
	return *beta != 0 && *ar < 0;
}

static void row_to_col(int n, const double *src, double *dst);
static void col_to_row(int n, const double *src, double *dst);

static lapack_int run_dgges(lapack_int n, double *a, double *b, lapack_int dosort,
	lapack_int *sdim, double *ar, double *ai, double *beta, double *vsl, double *vsr) {
	lapack_int info = LAPACKE_dgges(LAPACK_ROW_MAJOR, 'V', 'V', dosort ? 'S' : 'N',
		dosort ? select_negative : NULL, n, a, n, b, n, sdim, ar, ai, beta,
		vsl, n, vsr, n);
	return info;
}

extern void dtgex2_(int*, int*, int*, double*, int*, double*, int*, double*, int*,
	double*, int*, int*, int*, int*, double*, int*, int*);

static void row_to_col(int n, const double *src, double *dst) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) dst[j*n+i] = src[i*n+j];
}

static void col_to_row(int n, const double *src, double *dst) {
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) dst[i*n+j] = src[j*n+i];
}

static int run_dtgex2(int n, double *a, double *b, int j1, int n1, int n2) {
	double *ac = malloc((size_t)n*n*sizeof(double));
	double *bc = malloc((size_t)n*n*sizeof(double));
	double *q = calloc((size_t)n*n, sizeof(double));
	double *z = calloc((size_t)n*n, sizeof(double));
	int lwork = n*(n1+n2);
	if (lwork < 2*(n1+n2)*(n1+n2)) lwork = 2*(n1+n2)*(n1+n2);
	double *work = malloc((size_t)lwork*sizeof(double));
	for (int i = 0; i < n; i++) q[i*n+i] = z[i*n+i] = 1;
	row_to_col(n, a, ac);
	row_to_col(n, b, bc);
	int want = 1, lda = n, info = 0, fj1 = j1 + 1;
	dtgex2_(&want, &want, &n, ac, &lda, bc, &lda, q, &lda, z, &lda,
		&fj1, &n1, &n2, work, &lwork, &info);
	col_to_row(n, ac, a);
	col_to_row(n, bc, b);
	free(ac); free(bc); free(q); free(z); free(work);
	return info;
}

static lapack_int run_dtgexc(lapack_int n, double *a, double *b, lapack_int *ifst, lapack_int *ilst) {
	double *ac = malloc((size_t)n*n*sizeof(double));
	double *bc = malloc((size_t)n*n*sizeof(double));
	double *q = calloc((size_t)n*n, sizeof(double));
	double *z = calloc((size_t)n*n, sizeof(double));
	lapack_int lwork = n <= 1 ? 1 : 4*n + 16;
	double *work = malloc((size_t)lwork*sizeof(double));
	for (int i = 0; i < n; i++) q[i*n+i] = z[i*n+i] = 1;
	row_to_col(n, a, ac);
	row_to_col(n, b, bc);
	lapack_logical want = 1;
	lapack_int lda = n, info = 0;
	LAPACK_dtgexc(&want, &want, &n, ac, &lda, bc, &lda, q, &lda, z, &lda,
		ifst, ilst, work, &lwork, &info);
	col_to_row(n, ac, a);
	col_to_row(n, bc, b);
	free(ac); free(bc); free(q); free(z); free(work);
	return info;
}
*/
import "C"

import "unsafe"

func netlibDgges(n int, a, b []float64, dosort bool, ar, ai, beta, vsl, vsr []float64) (sdim, info int) {
	var csdim C.lapack_int
	cinfo := C.run_dgges(C.lapack_int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		C.lapack_int(boolInt(dosort)), &csdim, (*C.double)(unsafe.Pointer(&ar[0])),
		(*C.double)(unsafe.Pointer(&ai[0])), (*C.double)(unsafe.Pointer(&beta[0])),
		(*C.double)(unsafe.Pointer(&vsl[0])), (*C.double)(unsafe.Pointer(&vsr[0])))
	return int(csdim), int(cinfo)
}

func netlibDtgex2(n int, a, b []float64, j1, n1, n2 int) int {
	return int(C.run_dtgex2(C.int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		C.int(j1), C.int(n1), C.int(n2)))
}

func netlibDtgexc(n int, a, b []float64, ifst, ilst int) (ifstOut, ilstOut, info int) {
	cifst, cilst := C.lapack_int(ifst+1), C.lapack_int(ilst+1)
	cinfo := C.run_dtgexc(C.lapack_int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		&cifst, &cilst)
	return int(cifst) - 1, int(cilst) - 1, int(cinfo)
}
