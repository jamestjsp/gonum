// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f32

import (
	"simd/archsimd"
	"unsafe"
)

func ScalUnitary(alpha float32, x []float32) {
	a := archsimd.BroadcastFloat32x4(alpha)
	xData := unsafe.Pointer(unsafe.SliceData(x))
	i := 0
	for ; i+16 <= len(x); i += 16 {
		storeFloat32x4(xData, i, loadFloat32x4(xData, i).Mul(a))
		storeFloat32x4(xData, i+4, loadFloat32x4(xData, i+4).Mul(a))
		storeFloat32x4(xData, i+8, loadFloat32x4(xData, i+8).Mul(a))
		storeFloat32x4(xData, i+12, loadFloat32x4(xData, i+12).Mul(a))
	}
	for ; i+4 <= len(x); i += 4 {
		storeFloat32x4(xData, i, loadFloat32x4(xData, i).Mul(a))
	}
	for ; i < len(x); i++ {
		x[i] *= alpha
	}
}

func ScalUnitaryTo(dst []float32, alpha float32, x []float32) {
	if len(x) < 16 || len(dst) < len(x) || !simdDestinationCompatible(dst, x, len(x)) {
		scalUnitaryToScalar(dst, alpha, x)
		return
	}
	a := archsimd.BroadcastFloat32x4(alpha)
	dstData := unsafe.Pointer(unsafe.SliceData(dst))
	xData := unsafe.Pointer(unsafe.SliceData(x))
	i := 0
	for ; i+16 <= len(x); i += 16 {
		storeFloat32x4(dstData, i, loadFloat32x4(xData, i).Mul(a))
		storeFloat32x4(dstData, i+4, loadFloat32x4(xData, i+4).Mul(a))
		storeFloat32x4(dstData, i+8, loadFloat32x4(xData, i+8).Mul(a))
		storeFloat32x4(dstData, i+12, loadFloat32x4(xData, i+12).Mul(a))
	}
	for ; i+4 <= len(x); i += 4 {
		storeFloat32x4(dstData, i, loadFloat32x4(xData, i).Mul(a))
	}
	for ; i < len(x); i++ {
		dst[i] = alpha * x[i]
	}
}

func ScalInc(alpha float32, x []float32, n, incX uintptr) {
	var ix uintptr
	for i := 0; i < int(n); i++ {
		x[ix] *= alpha
		ix += incX
	}
}

func ScalIncTo(dst []float32, incDst uintptr, alpha float32, x []float32, n, incX uintptr) {
	var idst, ix uintptr
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha * x[ix]
		ix += incX
		idst += incDst
	}
}

func scalUnitaryToScalar(dst []float32, alpha float32, x []float32) {
	for i, v := range x {
		dst[i] = alpha * v
	}
}
