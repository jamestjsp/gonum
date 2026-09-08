// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha float64, x, y []float64) {
	if len(x) < 64 && !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() {
		if len(x) == 0 {
			return
		}
		y = y[:len(x)]
		if simdSlicesCompatible(x, y) {
			axpyShortNativeSIMD(len(x), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(y)), alpha)
			return
		}
	}
	axpyUnitaryPortableSIMD(alpha, x, y)
}
func AxpyUnitaryToSIMD(dst []float64, alpha float64, x, y []float64) {
	if len(x) < 64 && !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() {
		if len(x) == 0 {
			return
		}
		y = y[:len(x)]
		dst = dst[:len(x)]
		if simdSlicesCompatible(dst, x) && simdSlicesCompatible(dst, y) {
			axpyShortNativeSIMD(len(x), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(dst)), alpha)
			return
		}
	}
	axpyUnitaryToPortableSIMD(dst, alpha, x, y)
}
func axpyShortNativeSIMD(n int, xp, yp, dp unsafe.Pointer, alpha float64) {
	a := archsimd.BroadcastUint64x4(math.Float64bits(alpha)).AsFloat64x4()
	for n >= 16 {
		x, y, d := (*[16]float64)(xp), (*[16]float64)(yp), (*[16]float64)(dp)
		archsimd.LoadFloat64x4Array((*[4]float64)(x[0:4])).Mul(a).Add(archsimd.LoadFloat64x4Array((*[4]float64)(y[0:4]))).StoreArray((*[4]float64)(d[0:4]))
		archsimd.LoadFloat64x4Array((*[4]float64)(x[4:8])).Mul(a).Add(archsimd.LoadFloat64x4Array((*[4]float64)(y[4:8]))).StoreArray((*[4]float64)(d[4:8]))
		archsimd.LoadFloat64x4Array((*[4]float64)(x[8:12])).Mul(a).Add(archsimd.LoadFloat64x4Array((*[4]float64)(y[8:12]))).StoreArray((*[4]float64)(d[8:12]))
		archsimd.LoadFloat64x4Array((*[4]float64)(x[12:16])).Mul(a).Add(archsimd.LoadFloat64x4Array((*[4]float64)(y[12:16]))).StoreArray((*[4]float64)(d[12:16]))
		n -= 16
		if n == 0 {
			archsimd.ClearAVXUpperBits()
			return
		}
		xp, yp, dp = unsafe.Add(xp, 128), unsafe.Add(yp, 128), unsafe.Add(dp, 128)
	}
	for n >= 4 {
		archsimd.LoadFloat64x4Array((*[4]float64)(xp)).Mul(a).Add(archsimd.LoadFloat64x4Array((*[4]float64)(yp))).StoreArray((*[4]float64)(dp))
		n -= 4
		if n == 0 {
			archsimd.ClearAVXUpperBits()
			return
		}
		xp, yp, dp = unsafe.Add(xp, 32), unsafe.Add(yp, 32), unsafe.Add(dp, 32)
	}
	a2 := a.GetLo()
	if n >= 2 {
		archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Mul(a2).Add(archsimd.LoadFloat64x2Array((*[2]float64)(yp))).StoreArray((*[2]float64)(dp))
		n -= 2
		if n == 0 {
			archsimd.ClearAVXUpperBits()
			return
		}
		xp, yp, dp = unsafe.Add(xp, 16), unsafe.Add(yp, 16), unsafe.Add(dp, 16)
	}
	if n > 0 {
		var xv, yv archsimd.Uint64x2
		xv = xv.SetElem(0, *(*uint64)(xp))
		yv = yv.SetElem(0, *(*uint64)(yp))
		out := xv.AsFloat64x2().Mul(a2).Add(yv.AsFloat64x2()).ToBits().GetElem(0)
		*(*uint64)(dp) = out
	}
	archsimd.ClearAVXUpperBits()
}
