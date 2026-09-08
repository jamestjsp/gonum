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

// Share the complete eight-column input across rows. Bounds and disjointness
// checks precede every pointer access; dependent or descending inputs retain
// the checked sequential implementation.
func gerEightHardwareSIMD(m uintptr, alpha float64, x, y, a []float64, lda uintptr) bool {
	if m == 0 || m > uintptr(len(x)) || len(y) < 8 || len(a) < 8 || lda < 8 || (m > 1 && m-1 > uintptr(len(a)-8)/lda) || simd.Emulated() || simd.VectorBitSize() < 256 || !archsimd.X86.AVX2() || !simdMatrixDisjoint(a, x) || !simdMatrixDisjoint(a, y) {
		return false
	}
	gerEightNativeSIMD(m, alpha, unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(a)), lda*8)
	return true
}
func gerEightNativeSIMD(m uintptr, alpha float64, xp, yp, ap unsafe.Pointer, rowBytes uintptr) {
	av := archsimd.BroadcastUint64x4(math.Float64bits(alpha)).AsFloat64x4()
	y0 := archsimd.LoadFloat64x4Array((*[4]float64)(yp))
	y1 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, 32)))
	for m >= 4 {
		x0 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 0))).AsFloat64x4().Mul(av)
		a0 := unsafe.Add(ap, 0*rowBytes)
		y0.Mul(x0).Add(archsimd.LoadFloat64x4Array((*[4]float64)(a0))).StoreArray((*[4]float64)(a0))
		y1.Mul(x0).Add(archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a0, 32)))).StoreArray((*[4]float64)(unsafe.Add(a0, 32)))
		x1 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 8))).AsFloat64x4().Mul(av)
		a1 := unsafe.Add(ap, 1*rowBytes)
		y0.Mul(x1).Add(archsimd.LoadFloat64x4Array((*[4]float64)(a1))).StoreArray((*[4]float64)(a1))
		y1.Mul(x1).Add(archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a1, 32)))).StoreArray((*[4]float64)(unsafe.Add(a1, 32)))
		x2 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 16))).AsFloat64x4().Mul(av)
		a2 := unsafe.Add(ap, 2*rowBytes)
		y0.Mul(x2).Add(archsimd.LoadFloat64x4Array((*[4]float64)(a2))).StoreArray((*[4]float64)(a2))
		y1.Mul(x2).Add(archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a2, 32)))).StoreArray((*[4]float64)(unsafe.Add(a2, 32)))
		x3 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 24))).AsFloat64x4().Mul(av)
		a3 := unsafe.Add(ap, 3*rowBytes)
		y0.Mul(x3).Add(archsimd.LoadFloat64x4Array((*[4]float64)(a3))).StoreArray((*[4]float64)(a3))
		y1.Mul(x3).Add(archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a3, 32)))).StoreArray((*[4]float64)(unsafe.Add(a3, 32)))
		m -= 4
		if m == 0 {
			break
		}
		xp = unsafe.Add(xp, 32)
		ap = unsafe.Add(ap, 4*rowBytes)
	}
	for m > 0 {
		xv := archsimd.BroadcastUint64x4(*(*uint64)(xp)).AsFloat64x4().Mul(av)
		y0.Mul(xv).Add(archsimd.LoadFloat64x4Array((*[4]float64)(ap))).StoreArray((*[4]float64)(ap))
		y1.Mul(xv).Add(archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap, 32)))).StoreArray((*[4]float64)(unsafe.Add(ap, 32)))
		m--
		if m == 0 {
			break
		}
		xp = unsafe.Add(xp, 8)
		ap = unsafe.Add(ap, rowBytes)
	}
	archsimd.ClearAVXUpperBits()
}
