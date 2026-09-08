// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"simd"
	"simd/archsimd"
	"unsafe"
)

func complexAxpyShortSIMD(dst, x, y []complex64, alpha complex64) {
	n := len(x)
	y, dst = y[:n:n], dst[:n:n]
	xp, yp, dp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(dst))
	ar, ai := archsimd.BroadcastFloat32x8(real(alpha)), archsimd.BroadcastFloat32x8(imag(alpha))
	var offset uintptr
	for ; n >= 8; n -= 8 {
		x0 := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, offset)))
		x1 := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, offset+32)))
		y0 := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, offset)))
		y1 := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, offset+32)))
		x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalarsGrouped(1, 0, 7, 6, x0).Mul(ai)).Add(y0).StoreArray((*[8]float32)(unsafe.Add(dp, offset)))
		x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalarsGrouped(1, 0, 7, 6, x1).Mul(ai)).Add(y1).StoreArray((*[8]float32)(unsafe.Add(dp, offset+32)))
		offset += 64
	}
	if n >= 4 {
		xv := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, offset)))
		yv := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, offset)))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalarsGrouped(1, 0, 7, 6, xv).Mul(ai)).Add(yv).StoreArray((*[8]float32)(unsafe.Add(dp, offset)))
		offset += 32
		n -= 4
	}
	if n >= 2 {
		xv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, offset)))
		yv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(yp, offset)))
		var result archsimd.Float32x4
		if simd.VectorBitSize() == 256 || len(x)&7 < 4 {
			// The established kernel handles this pair with Go scalar
			// complex64 products, which compute in float64 and round the
			// product to float32 before adding y.
			xw := xv.ConvertToFloat64()
			result = xw.Mul(ar.GetLo().ConvertToFloat64()).AddOddSubEven(xw.ConcatPermuteScalarsGrouped(1, 2, xw).Mul(ai.GetLo().ConvertToFloat64())).ConvertToFloat32().Add(yv)
		} else {
			result = xv.Mul(ar.GetLo()).AddOddSubEven(xv.ConcatPermuteScalars(1, 0, 7, 6, xv).Mul(ai.GetLo())).Add(yv)
		}
		result.StoreArray((*[4]float32)(unsafe.Add(dp, offset)))
		offset += 16
		n -= 2
	}
	if n != 0 {
		var xv, yv archsimd.Uint64x2
		x0 := xv.SetElem(0, *(*uint64)(unsafe.Add(xp, offset))).AsFloat32x4()
		y0 := yv.SetElem(0, *(*uint64)(unsafe.Add(yp, offset))).AsFloat32x4()
		// Every established odd tail uses widened scalar multiplication.
		xw := x0.ConvertToFloat64()
		r := xw.Mul(ar.GetLo().ConvertToFloat64()).AddOddSubEven(xw.ConcatPermuteScalarsGrouped(1, 2, xw).Mul(ai.GetLo().ConvertToFloat64())).ConvertToFloat32().Add(y0).AsUint64x2()
		*(*uint64)(unsafe.Add(dp, offset)) = r.GetElem(0)
	}
}
