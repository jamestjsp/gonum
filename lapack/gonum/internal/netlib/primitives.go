// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package netlib

/*
#cgo CFLAGS: -I/opt/homebrew/opt/lapack/include
#cgo LDFLAGS: -L/opt/homebrew/opt/lapack/lib -Wl,-rpath,/opt/homebrew/opt/lapack/lib -llapack
#include <lapack.h>

extern void dlagv2_(double*, lapack_int*, double*, lapack_int*, double*,
	double*, double*, double*, double*, double*, double*);
extern void dlag2_(double*, lapack_int*, double*, lapack_int*, double*, double*,
	double*, double*, double*, double*);
extern void dlartg_(double*, double*, double*, double*, double*);
extern void dlasv2_(double*, double*, double*, double*, double*, double*,
	double*, double*, double*);

static double run_dlapy2(double x, double y) {
	return LAPACK_dlapy2(&x, &y);
}

static double run_dlange(char norm, lapack_int m, lapack_int n, double *a,
		lapack_int lda, double *work) {
	return LAPACK_dlange(&norm, &m, &n, a, &lda, work);
}

static void run_dlagv2(double *a, lapack_int lda, double *b, lapack_int ldb,
		double *alphar, double *alphai, double *beta, double *csl, double *snl,
		double *csr, double *snr) {
	dlagv2_(a, &lda, b, &ldb, alphar, alphai, beta, csl, snl, csr, snr);
}

static void run_dlag2(double *a, lapack_int lda, double *b, lapack_int ldb,
		double safmin, double *scale1, double *scale2, double *wr1, double *wr2,
		double *wi) {
	dlag2_(a, &lda, b, &ldb, &safmin, scale1, scale2, wr1, wr2, wi);
}

static void run_dlartg(double f, double g, double *cs, double *sn, double *r) {
	dlartg_(&f, &g, cs, sn, r);
}

static void run_dlasv2(double f, double g, double h, double *ssmin,
		double *ssmax, double *snr, double *csr, double *snl, double *csl) {
	dlasv2_(&f, &g, &h, ssmin, ssmax, snr, csr, snl, csl);
}

static void run_dlassq(lapack_int n, double *x, lapack_int incx,
		double *scale, double *sumsq) {
	LAPACK_dlassq(&n, x, &incx, scale, sumsq);
}

static double run_dlanhs(char norm, lapack_int n, double *a, lapack_int lda,
		double *work) {
	return LAPACK_dlanhs(&norm, &n, a, &lda, work);
}
*/
import "C"

func Dlapy2(x, y float64) float64 {
	return float64(C.run_dlapy2(C.double(x), C.double(y)))
}

// Dlange computes a matrix norm of a column-major matrix.
func Dlange(norm byte, m, n int, a []float64, lda int, work []float64) float64 {
	return float64(C.run_dlange(C.char(norm), C.lapack_int(m), C.lapack_int(n),
		doublePtr(a), C.lapack_int(lda), doublePtr(work)))
}

// Dlagv2 computes the generalized Schur factorization of a column-major 2×2 pencil.
func Dlagv2(a []float64, lda int, b []float64, ldb int) (alphar, alphai, beta [2]float64, csl, snl, csr, snr float64) {
	var ccsl, csnl, ccsr, csnr C.double
	C.run_dlagv2(doublePtr(a), C.lapack_int(lda), doublePtr(b), C.lapack_int(ldb),
		doublePtr(alphar[:]), doublePtr(alphai[:]), doublePtr(beta[:]),
		&ccsl, &csnl, &ccsr, &csnr)
	csl, snl, csr, snr = float64(ccsl), float64(csnl), float64(ccsr), float64(csnr)
	return alphar, alphai, beta, csl, snl, csr, snr
}

func Dlag2(a []float64, lda int, b []float64, ldb int, safmin float64) (scale1, scale2, wr1, wr2, wi float64) {
	var cs1, cs2, cwr1, cwr2, cwi C.double
	C.run_dlag2(doublePtr(a), C.lapack_int(lda), doublePtr(b), C.lapack_int(ldb), C.double(safmin),
		&cs1, &cs2, &cwr1, &cwr2, &cwi)
	return float64(cs1), float64(cs2), float64(cwr1), float64(cwr2), float64(cwi)
}

func Dlartg(f, g float64) (cs, sn, r float64) {
	var ccs, csn, cr C.double
	C.run_dlartg(C.double(f), C.double(g), &ccs, &csn, &cr)
	return float64(ccs), float64(csn), float64(cr)
}

func Dlasv2(f, g, h float64) (ssmin, ssmax, snr, csr, snl, csl float64) {
	var cmin, cmax, crs, crc, cls, clc C.double
	C.run_dlasv2(C.double(f), C.double(g), C.double(h), &cmin, &cmax, &crs, &crc, &cls, &clc)
	return float64(cmin), float64(cmax), float64(crs), float64(crc), float64(cls), float64(clc)
}

func Dlassq(n int, x []float64, incx int, scale, sumsq float64) (float64, float64) {
	cscale, csumsq := C.double(scale), C.double(sumsq)
	C.run_dlassq(C.lapack_int(n), doublePtr(x), C.lapack_int(incx), &cscale, &csumsq)
	return float64(cscale), float64(csumsq)
}

// Dlanhs computes a norm of a column-major upper Hessenberg matrix.
func Dlanhs(norm byte, n int, a []float64, lda int, work []float64) float64 {
	return float64(C.run_dlanhs(C.char(norm), C.lapack_int(n), doublePtr(a),
		C.lapack_int(lda), doublePtr(work)))
}
