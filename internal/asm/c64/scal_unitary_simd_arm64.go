// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package c64

import "simd/archsimd"

func scalUnitary(alpha complex64, x []complex64) {
	if len(x) < 8 {
		for i := range x {
			x[i] *= alpha
		}
		return
	}
	ar, ai := complexAlphaVectors(alpha)
	var i int
	for ; i+2 <= len(x); i += 2 {
		complexScaleVector(loadComplex64x2(x, i), ar, ai).StoreArray(complex64Array(x, i))
	}
	if i < len(x) {
		x[i] *= alpha
	}
}

func scalUnitaryTo(dst []complex64, alpha complex64, x []complex64) {
	if len(dst) < len(x) || !simdSlicesCompatible(dst[:len(x)], x) || len(x) < 8 {
		for i, v := range x {
			dst[i] = alpha * v
		}
		return
	}
	ar, ai := complexAlphaVectors(alpha)
	var i int
	for ; i+2 <= len(x); i += 2 {
		complexScaleVector(loadComplex64x2(x, i), ar, ai).StoreArray(complex64Array(dst, i))
	}
	if i < len(x) {
		dst[i] = alpha * x[i]
	}
}

func sscalUnitary(alpha float32, x []complex64) {
	a := archsimd.BroadcastFloat32x4(alpha)
	var i int
	for ; i+2 <= len(x); i += 2 {
		loadComplex64x2(x, i).Mul(a).StoreArray(complex64Array(x, i))
	}
	if i < len(x) {
		v := x[i]
		x[i] = complex(real(v)*alpha, imag(v)*alpha)
	}
}
