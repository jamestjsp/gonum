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

// Each block shares x across four row dots, and finishes all arithmetic in
// vector registers. The public caller resumes unstored rows on recovery.
func gemvNBlockHardwareSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, beta float64, y []float64) (uintptr, bool) {
	if m < 4 || n < 4 || uintptr(len(x)) < n || uintptr(len(y)) < m || uintptr(len(a)) < n || lda < n || !simdPositiveSpan(len(a)-int(n)+1, m, lda, 0) || simd.Emulated() || simd.VectorBitSize() < 256 || !archsimd.X86.AVX2() || !simdMatrixDisjoint(y, x) || !simdMatrixDisjoint(y, a) {
		return 0, false
	}
	av := archsimd.BroadcastUint64x4(math.Float64bits(alpha)).AsFloat64x4()
	bv := archsimd.BroadcastUint64x4(math.Float64bits(beta)).AsFloat64x4()
	inf := archsimd.BroadcastFloat64x4(math.Inf(1))
	sign := archsimd.BroadcastUint64x4(0x8000000000000000)
	betaZero := math.Float64bits(beta)<<1 == 0
	ap, yp := unsafe.Pointer(unsafe.SliceData(a)), unsafe.Pointer(unsafe.SliceData(y))
	row := uintptr(0)
	for ; row+4 <= m; row += 4 {
		p0, p1, p2, p3 := unsafe.Add(ap, row*lda*8), unsafe.Add(ap, (row+1)*lda*8), unsafe.Add(ap, (row+2)*lda*8), unsafe.Add(ap, (row+3)*lda*8)
		xp := unsafe.Pointer(unsafe.SliceData(x))
		var d0, d1, d2, d3 archsimd.Float64x4
		rem := n
		for rem >= 4 {
			xv := archsimd.LoadFloat64x4Array((*[4]float64)(xp))
			d0 = archsimd.LoadFloat64x4Array((*[4]float64)(p0)).Mul(xv).Add(d0)
			d1 = archsimd.LoadFloat64x4Array((*[4]float64)(p1)).Mul(xv).Add(d1)
			d2 = archsimd.LoadFloat64x4Array((*[4]float64)(p2)).Mul(xv).Add(d2)
			d3 = archsimd.LoadFloat64x4Array((*[4]float64)(p3)).Mul(xv).Add(d3)
			rem -= 4
			if rem == 0 {
				break
			}
			xp, p0, p1, p2, p3 = unsafe.Add(xp, 32), unsafe.Add(p0, 32), unsafe.Add(p1, 32), unsafe.Add(p2, 32), unsafe.Add(p3, 32)
		}
		if rem > 0 {
			xv := gemvLoadTailSIMD(xp, rem)
			d0 = gemvLoadTailSIMD(p0, rem).Mul(xv).Add(d0)
			d1 = gemvLoadTailSIMD(p1, rem).Mul(xv).Add(d1)
			d2 = gemvLoadTailSIMD(p2, rem).Mul(xv).Add(d2)
			d3 = gemvLoadTailSIMD(p3, rem).Mul(xv).Add(d3)
		}
		h01, h23 := d0.ConcatAddPairsGrouped(d1), d2.ConcatAddPairsGrouped(d3)
		var values archsimd.Float64x4
		values = values.SetLo(h01.GetLo().Add(h01.GetHi())).SetHi(h23.GetLo().Add(h23.GetHi())).Mul(av)
		out := (*[4]float64)(unsafe.Add(yp, row*8))
		if !betaZero {
			values = values.Add(archsimd.LoadFloat64x4Array(out).Mul(bv))
		}
		if values.ToBits().AndNot(sign).AsFloat64x4().Less(inf).ToBits() != 15 {
			archsimd.ClearAVXUpperBits()
			return row, false
		}
		values.StoreArray(out)
	}
	archsimd.ClearAVXUpperBits()
	return row, row == m
}
func gemvLoadTailSIMD(p unsafe.Pointer, n uintptr) archsimd.Float64x4 {
	var lo archsimd.Uint64x2
	var hi archsimd.Uint64x2
	var high uint64
	lo = lo.SetElem(0, *(*uint64)(p))
	if n > 1 {
		lo = lo.SetElem(1, *(*uint64)(unsafe.Add(p, 8)))
	}
	if n > 2 {
		high = *(*uint64)(unsafe.Add(p, 16))
	}
	hi = hi.SetElem(0, high)
	var v archsimd.Uint64x4
	return v.SetLo(lo).SetHi(hi).AsFloat64x4()
}
