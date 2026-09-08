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

// Validate before scaling y: overlapping inputs retain the original sequential
// updates, in which the scaling itself can change later matrix/vector inputs.
func gemvTEightHardwareSIMD(m uintptr, alpha float64, a []float64, lda uintptr, x []float64, beta float64, y []float64) bool {
	if m == 0 || m > uintptr(len(x)) || len(y) < 8 || len(a) < 8 || lda < 8 || (m > 1 && m-1 > uintptr(len(a)-8)/lda) || simd.Emulated() || simd.VectorBitSize() < 256 || !archsimd.X86.AVX2() || !simdMatrixDisjoint(y, x) || !simdMatrixDisjoint(y, a) {
		return false
	}
	gemvTEightNativeSIMD(m, math.Float64bits(alpha), math.Float64bits(beta), unsafe.Pointer(unsafe.SliceData(a)), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), lda*8)
	return true
}

// Keep both output vectors live through every row. Multiplication and addition
// remain separate and each output follows the original row-update order.
func gemvTEightNativeSIMD(m uintptr, alphaBits, betaBits uint64, ap, xp, yp unsafe.Pointer, rowBytes uintptr) {
	av := archsimd.BroadcastUint64x4(alphaBits).AsFloat64x4()
	var y0, y1 archsimd.Float64x4
	if betaBits<<1 != 0 {
		bv := archsimd.BroadcastUint64x4(betaBits).AsFloat64x4()
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(yp)).Mul(bv)
		y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, 32))).Mul(bv)
	}
	for m >= 4 {
		x0 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 0))).AsFloat64x4().Mul(av)
		x1 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 8))).AsFloat64x4().Mul(av)
		x2 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 16))).AsFloat64x4().Mul(av)
		x3 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 24))).AsFloat64x4().Mul(av)
		a0 := unsafe.Add(ap, 0*rowBytes)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(a0)).Mul(x0).Add(y0)
		y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a0, 32))).Mul(x0).Add(y1)
		a1 := unsafe.Add(ap, 1*rowBytes)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(a1)).Mul(x1).Add(y0)
		y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a1, 32))).Mul(x1).Add(y1)
		a2 := unsafe.Add(ap, 2*rowBytes)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(a2)).Mul(x2).Add(y0)
		y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a2, 32))).Mul(x2).Add(y1)
		a3 := unsafe.Add(ap, 3*rowBytes)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(a3)).Mul(x3).Add(y0)
		y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(a3, 32))).Mul(x3).Add(y1)
		m -= 4
		if m == 0 {
			break
		}
		xp = unsafe.Add(xp, 32)
		ap = unsafe.Add(ap, 4*rowBytes)
	}
	for m > 0 {
		xv := archsimd.BroadcastUint64x4(*(*uint64)(xp)).AsFloat64x4().Mul(av)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(ap)).Mul(xv).Add(y0)
		y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap, 32))).Mul(xv).Add(y1)
		m--
		if m == 0 {
			break
		}
		xp = unsafe.Add(xp, 8)
		ap = unsafe.Add(ap, rowBytes)
	}
	y0.StoreArray((*[4]float64)(yp))
	y1.StoreArray((*[4]float64)(unsafe.Add(yp, 32)))
	archsimd.ClearAVXUpperBits()
}
