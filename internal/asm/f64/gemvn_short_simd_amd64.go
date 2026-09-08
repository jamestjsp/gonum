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

// Four independent row dots reuse both complete x vectors. The checked
// public path handles other dimensions, aliases and descending strides.
func gemvNShortHardwareSIMD(m uintptr, alpha float64, a []float64, lda uintptr, x []float64, beta float64, y []float64) (uintptr, bool) {
	if m < 4 || len(x) < 8 || uintptr(len(y)) < m || len(a) < 8 || lda < 8 || !simdPositiveSpan(len(a)-7, m, lda, 0) || simd.Emulated() || simd.VectorBitSize() < 256 || !archsimd.X86.AVX2() || !simdMatrixDisjoint(y, x) || !simdMatrixDisjoint(y, a) {
		return 0, false
	}
	xv0 := archsimd.LoadFloat64x4Array((*[4]float64)(x[:4]))
	xv1 := archsimd.LoadFloat64x4Array((*[4]float64)(x[4:8]))
	av := archsimd.BroadcastUint64x4(math.Float64bits(alpha)).AsFloat64x4()
	bv := archsimd.BroadcastUint64x4(math.Float64bits(beta)).AsFloat64x4()
	betaZero := math.Float64bits(beta)<<1 == 0
	ap := unsafe.Pointer(unsafe.SliceData(a))
	yp := unsafe.Pointer(unsafe.SliceData(y))
	row := uintptr(0)
	mask := archsimd.BroadcastUint64x4(0x8000000000000000)
	inf := archsimd.BroadcastFloat64x4(math.Inf(1))
	for ; row+4 <= m; row += 4 {
		r0 := (*[8]float64)(unsafe.Add(ap, (row+0)*lda*8))
		d0 := archsimd.LoadFloat64x4Array((*[4]float64)(r0[:4])).Mul(xv0).Add(archsimd.LoadFloat64x4Array((*[4]float64)(r0[4:8])).Mul(xv1))
		r1 := (*[8]float64)(unsafe.Add(ap, (row+1)*lda*8))
		d1 := archsimd.LoadFloat64x4Array((*[4]float64)(r1[:4])).Mul(xv0).Add(archsimd.LoadFloat64x4Array((*[4]float64)(r1[4:8])).Mul(xv1))
		r2 := (*[8]float64)(unsafe.Add(ap, (row+2)*lda*8))
		d2 := archsimd.LoadFloat64x4Array((*[4]float64)(r2[:4])).Mul(xv0).Add(archsimd.LoadFloat64x4Array((*[4]float64)(r2[4:8])).Mul(xv1))
		r3 := (*[8]float64)(unsafe.Add(ap, (row+3)*lda*8))
		d3 := archsimd.LoadFloat64x4Array((*[4]float64)(r3[:4])).Mul(xv0).Add(archsimd.LoadFloat64x4Array((*[4]float64)(r3[4:8])).Mul(xv1))
		h01 := d0.ConcatAddPairsGrouped(d1)
		h23 := d2.ConcatAddPairsGrouped(d3)
		var values archsimd.Float64x4
		values = values.SetLo(h01.GetLo().Add(h01.GetHi())).SetHi(h23.GetLo().Add(h23.GetHi())).Mul(av)
		if !betaZero {
			values = values.Add(archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, row*8))).Mul(bv))
		}
		if values.ToBits().AndNot(mask).AsFloat64x4().Less(inf).ToBits() != 15 {
			archsimd.ClearAVXUpperBits()
			return row, false
		}
		values.StoreArray((*[4]float64)(unsafe.Add(yp, row*8)))
	}
	archsimd.ClearAVXUpperBits()
	return row, row == m
}
