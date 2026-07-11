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

static lapack_logical select_large_alpha(const double *ar, const double *ai, const double *beta) {
	(void)ai;
	(void)beta;
	return *ar > 1e-299;
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

static lapack_int run_dgges_large_alpha(lapack_int n, double *a, double *b,
		lapack_int *sdim, double *ar, double *ai, double *beta, double *vsl, double *vsr) {
	return LAPACKE_dgges(LAPACK_ROW_MAJOR, 'V', 'V', 'S', select_large_alpha,
		n, a, n, b, n, sdim, ar, ai, beta, vsl, n, vsr, n);
}

static lapack_int query_dtgsen(lapack_int ijob, lapack_int n, lapack_int lwork,
		lapack_int liwork, double *work_out, lapack_int *iwork_out) {
	lapack_logical select[2] = {1, 0};
	double a[4] = {1, 0, 0, 2};
	double b[4] = {1, 0, 0, 1};
	double ar[2] = {0, 0}, ai[2] = {0, 0}, beta[2] = {0, 0};
	double q[4] = {1, 0, 0, 1}, z[4] = {1, 0, 0, 1};
	double dif[2] = {0, 0}, work[1] = {-1}, pl = 0, pr = 0;
	lapack_int iwork[1] = {-1}, m = 0;
	lapack_int ld = n > 0 ? n : 1;
	lapack_int info = LAPACKE_dtgsen_work(LAPACK_ROW_MAJOR, ijob, 0, 0,
		select, n, a, ld, b, ld, ar, ai, beta, q, ld, z, ld, &m, &pl, &pr,
		dif, work, lwork, iwork, liwork);
	*work_out = work[0];
	*iwork_out = iwork[0];
	return info;
}

static lapack_int run_dhgeqz(char job, char compq, char compz, lapack_int n,
		lapack_int ilo, lapack_int ihi, double *h, double *t, double *ar,
		double *ai, double *beta, double *q, double *z) {
	lapack_int lwork = n > 0 ? n : 1;
	double *work = malloc((size_t)lwork*sizeof(double));
	lapack_int info = LAPACKE_dhgeqz_work(LAPACK_ROW_MAJOR, job, compq, compz,
		n, ilo, ihi, h, n, t, n, ar, ai, beta, q, n, z, n, work, lwork);
	free(work);
	return info;
}

static lapack_int run_dtgsen_singular(void) {
	lapack_logical select[2] = {1, 0};
	double a[4] = {1, 0, 0, 1};
	double b[4] = {1, 0, 0, 1};
	double ar[2] = {0, 0}, ai[2] = {0, 0}, beta[2] = {0, 0};
	double q[4] = {1, 0, 0, 1}, z[4] = {1, 0, 0, 1};
	double dif[2] = {0, 0}, pl = 0, pr = 0;
	lapack_int m = 0;
	return LAPACKE_dtgsen(LAPACK_ROW_MAJOR, 1, 0, 0, select, 2, a, 2, b, 2,
		ar, ai, beta, q, 2, z, 2, &m, &pl, &pr, dif);
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

func netlibDggesLargeAlpha(n int, a, b []float64, ar, ai, beta, vsl, vsr []float64) (sdim, info int) {
	var csdim C.lapack_int
	cinfo := C.run_dgges_large_alpha(C.lapack_int(n), (*C.double)(unsafe.Pointer(&a[0])), (*C.double)(unsafe.Pointer(&b[0])),
		&csdim, (*C.double)(unsafe.Pointer(&ar[0])), (*C.double)(unsafe.Pointer(&ai[0])),
		(*C.double)(unsafe.Pointer(&beta[0])), (*C.double)(unsafe.Pointer(&vsl[0])),
		(*C.double)(unsafe.Pointer(&vsr[0])))
	return int(csdim), int(cinfo)
}

func netlibDtgsenWorkspace(ijob, n, lwork, liwork int) (work float64, iwork, info int) {
	var cwork C.double
	var ciwork C.lapack_int
	cinfo := C.query_dtgsen(C.lapack_int(ijob), C.lapack_int(n), C.lapack_int(lwork), C.lapack_int(liwork),
		&cwork, &ciwork)
	return float64(cwork), int(ciwork), int(cinfo)
}

func netlibDhgeqz(job, compq, compz byte, n, ilo, ihi int, h, t, ar, ai, beta, q, z []float64) int {
	return int(C.run_dhgeqz(C.char(job), C.char(compq), C.char(compz), C.lapack_int(n),
		C.lapack_int(ilo+1), C.lapack_int(ihi+1), (*C.double)(unsafe.Pointer(&h[0])),
		(*C.double)(unsafe.Pointer(&t[0])), (*C.double)(unsafe.Pointer(&ar[0])),
		(*C.double)(unsafe.Pointer(&ai[0])), (*C.double)(unsafe.Pointer(&beta[0])),
		(*C.double)(unsafe.Pointer(&q[0])), (*C.double)(unsafe.Pointer(&z[0]))))
}

func netlibDtgsenSingular() int {
	return int(C.run_dtgsen_singular())
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
