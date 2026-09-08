// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"simd"
	"simd/archsimd"
	"unsafe"
)

func DivSIMD(dst, src []float64) {
	if len(src) == 0 {
		return
	}
	d := dst[:len(src)]
	if !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() && simdSlicesCompatible(d, src) {
		divShortNativeSIMD(d, d, src)
		return
	}
	divPortableSIMD(dst, src)
}
func DivToSIMD(dst, x, y []float64) []float64 {
	if len(x) == 0 {
		return dst
	}
	d := dst[:len(x)]
	y = y[:len(x)]
	if !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() && simdSlicesCompatible(d, x) && simdSlicesCompatible(d, y) {
		divShortNativeSIMD(d, x, y)
		return dst
	}
	return divToPortableSIMD(dst, x, y)
}
func divShortNativeSIMD(dst, x, y []float64) {
	n := len(x)
	xp, yp, dp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(dst))
	for n >= 16 {
		xb, yb, db := (*[16]float64)(xp), (*[16]float64)(yp), (*[16]float64)(dp)
		archsimd.LoadFloat64x4Array((*[4]float64)(xb[:4])).Div(archsimd.LoadFloat64x4Array((*[4]float64)(yb[:4]))).StoreArray((*[4]float64)(db[:4]))
		archsimd.LoadFloat64x4Array((*[4]float64)(xb[4:8])).Div(archsimd.LoadFloat64x4Array((*[4]float64)(yb[4:8]))).StoreArray((*[4]float64)(db[4:8]))
		archsimd.LoadFloat64x4Array((*[4]float64)(xb[8:12])).Div(archsimd.LoadFloat64x4Array((*[4]float64)(yb[8:12]))).StoreArray((*[4]float64)(db[8:12]))
		archsimd.LoadFloat64x4Array((*[4]float64)(xb[12:16])).Div(archsimd.LoadFloat64x4Array((*[4]float64)(yb[12:16]))).StoreArray((*[4]float64)(db[12:16]))
		n -= 16
		if n == 0 {
			archsimd.ClearAVXUpperBits()
			return
		}
		xp, yp, dp = unsafe.Add(xp, 128), unsafe.Add(yp, 128), unsafe.Add(dp, 128)
	}
	for n >= 4 {
		archsimd.LoadFloat64x4Array((*[4]float64)(xp)).Div(archsimd.LoadFloat64x4Array((*[4]float64)(yp))).StoreArray((*[4]float64)(dp))
		n -= 4
		if n == 0 {
			archsimd.ClearAVXUpperBits()
			return
		}
		xp, yp, dp = unsafe.Add(xp, 32), unsafe.Add(yp, 32), unsafe.Add(dp, 32)
	}
	if n >= 2 {
		archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Div(archsimd.LoadFloat64x2Array((*[2]float64)(yp))).StoreArray((*[2]float64)(dp))
		n -= 2
		if n == 0 {
			archsimd.ClearAVXUpperBits()
			return
		}
		xp, yp, dp = unsafe.Add(xp, 16), unsafe.Add(yp, 16), unsafe.Add(dp, 16)
	}
	if n > 0 {
		var a, b archsimd.Uint64x2
		a = a.SetElem(0, *(*uint64)(xp))
		b = b.SetElem(0, *(*uint64)(yp)).SetElem(1, 0x3ff0000000000000)
		bits := a.AsFloat64x2().Div(b.AsFloat64x2()).ToBits().GetElem(0)
		*(*uint64)(dp) = bits
	}
	archsimd.ClearAVXUpperBits()
}
