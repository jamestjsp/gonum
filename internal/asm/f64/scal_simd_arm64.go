// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import "simd/archsimd"

// ScalUnitary is
//
//	for i := range x {
//		x[i] *= alpha
//	}
func ScalUnitary(alpha float64, x []float64) {
	a := archsimd.BroadcastFloat64x2(alpha)
	i := 0
	for ; i+8 <= len(x); i += 8 {
		xi := x[i : i+8]
		x0 := archsimd.LoadFloat64x2(xi)
		x1 := archsimd.LoadFloat64x2(xi[2:])
		x2 := archsimd.LoadFloat64x2(xi[4:])
		x3 := archsimd.LoadFloat64x2(xi[6:])

		x0.Mul(a).Store(xi)
		x1.Mul(a).Store(xi[2:])
		x2.Mul(a).Store(xi[4:])
		x3.Mul(a).Store(xi[6:])
	}
	for ; i+2 <= len(x); i += 2 {
		xi := x[i : i+2]
		archsimd.LoadFloat64x2(xi).Mul(a).Store(xi)
	}
	if i < len(x) {
		x[i] *= alpha
	}
}

// ScalUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha * v
//	}
func ScalUnitaryTo(dst []float64, alpha float64, x []float64) {
	a := archsimd.BroadcastFloat64x2(alpha)
	i := 0
	for ; i+8 <= len(x); i += 8 {
		xi := x[i : i+8]
		di := dst[i : i+8]
		x0 := archsimd.LoadFloat64x2(xi)
		x1 := archsimd.LoadFloat64x2(xi[2:])
		x2 := archsimd.LoadFloat64x2(xi[4:])
		x3 := archsimd.LoadFloat64x2(xi[6:])

		x0.Mul(a).Store(di)
		x1.Mul(a).Store(di[2:])
		x2.Mul(a).Store(di[4:])
		x3.Mul(a).Store(di[6:])
	}
	for ; i+2 <= len(x); i += 2 {
		xi := x[i : i+2]
		di := dst[i : i+2]
		archsimd.LoadFloat64x2(xi).Mul(a).Store(di)
	}
	if i < len(x) {
		dst[i] = alpha * x[i]
	}
}

// ScalInc is
//
//	var ix uintptr
//	for i := 0; i < int(n); i++ {
//		x[ix] *= alpha
//		ix += incX
//	}
func ScalInc(alpha float64, x []float64, n, incX uintptr) {
	var ix uintptr
	for i := 0; i < int(n); i++ {
		x[ix] *= alpha
		ix += incX
	}
}

// ScalIncTo is
//
//	var idst, ix uintptr
//	for i := 0; i < int(n); i++ {
//		dst[idst] = alpha * x[ix]
//		ix += incX
//		idst += incDst
//	}
func ScalIncTo(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr) {
	var idst, ix uintptr
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha * x[ix]
		ix += incX
		idst += incDst
	}
}
