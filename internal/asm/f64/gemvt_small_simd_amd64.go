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

func gemvTSmallHardwareSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, beta float64, y []float64) bool {
	if n < 4 || n > 16 || m == 0 || m > uintptr(len(x)) || n > uintptr(len(y)) || n > uintptr(len(a)) || lda < n || (m > 1 && m-1 > (uintptr(len(a))-n)/lda) || simd.Emulated() || simd.VectorBitSize() < 256 || !archsimd.X86.AVX2() || !simdMatrixDisjoint(y, x) || !simdMatrixDisjoint(y, a) {
		return false
	}
	gemvTSmallNativeSIMD(m, n, math.Float64bits(alpha), math.Float64bits(beta), unsafe.Pointer(unsafe.SliceData(a)), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), lda*8)
	return true
}

// Each active output stays in a register through the original sequence of row
// updates. The final partial block uses exact two-element and one-element loads.
func gemvTSmallNativeSIMD(m, n uintptr, alphaBits, betaBits uint64, ap, xp, yp unsafe.Pointer, rowBytes uintptr) {
	av := archsimd.BroadcastUint64x4(alphaBits).AsFloat64x4()
	var y0, y1, y2, y3 archsimd.Float64x4
	var pairLo, pairHi, oddBits uint64
	tailScaleBits := uint64(0x3ff0000000000000)
	pairOffset := (n &^ 3) * 8
	oddOffset := (n - 1) * 8
	if betaBits<<1 != 0 {
		bv := archsimd.BroadcastUint64x4(betaBits).AsFloat64x4()
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(yp)).Mul(bv)
		if n >= 8 {
			y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, 32))).Mul(bv)
		}
		if n >= 12 {
			y2 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, 64))).Mul(bv)
		}
		if n == 16 {
			y3 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, 96))).Mul(bv)
		}
		if n&2 != 0 {
			p := (*[2]uint64)(unsafe.Add(yp, pairOffset))
			pairLo, pairHi = p[0], p[1]
		}
		if n&1 != 0 {
			oddBits = *(*uint64)(unsafe.Add(yp, oddOffset))
		}
		tailScaleBits = betaBits
	}
	// Integer state crosses the beta branch; constructing the two-lane
	// values here avoids legacy XMM merge copies with live YMM outputs.
	var pairBits archsimd.Uint64x2
	pairBits = pairBits.SetElem(0, pairLo).SetElem(1, pairHi)
	tailScale := archsimd.BroadcastUint64x2(tailScaleBits).AsFloat64x2()
	pair := pairBits.AsFloat64x2().Mul(tailScale)
	odd := archsimd.BroadcastUint64x2(oddBits).AsFloat64x2().Mul(tailScale)
	// Prepare independent row products before applying their original-order sums.
	// Each pointer names a consumed row; no unused one-past-end pointer is formed.
	for m >= 4 {
		xv0 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 0))).AsFloat64x4().Mul(av)
		ap0 := unsafe.Add(ap, 0*rowBytes)
		xv1 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 8))).AsFloat64x4().Mul(av)
		ap1 := unsafe.Add(ap, 1*rowBytes)
		xv2 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 16))).AsFloat64x4().Mul(av)
		ap2 := unsafe.Add(ap, 2*rowBytes)
		xv3 := archsimd.BroadcastUint64x4(*(*uint64)(unsafe.Add(xp, 24))).AsFloat64x4().Mul(av)
		ap3 := unsafe.Add(ap, 3*rowBytes)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap0, 0))).Mul(xv0).Add(y0)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap1, 0))).Mul(xv1).Add(y0)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap2, 0))).Mul(xv2).Add(y0)
		y0 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap3, 0))).Mul(xv3).Add(y0)
		if n >= 8 {
			y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap0, 32))).Mul(xv0).Add(y1)
			y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap1, 32))).Mul(xv1).Add(y1)
			y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap2, 32))).Mul(xv2).Add(y1)
			y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap3, 32))).Mul(xv3).Add(y1)
		}
		if n >= 12 {
			y2 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap0, 64))).Mul(xv0).Add(y2)
			y2 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap1, 64))).Mul(xv1).Add(y2)
			y2 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap2, 64))).Mul(xv2).Add(y2)
			y2 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap3, 64))).Mul(xv3).Add(y2)
		}
		if n == 16 {
			y3 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap0, 96))).Mul(xv0).Add(y3)
			y3 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap1, 96))).Mul(xv1).Add(y3)
			y3 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap2, 96))).Mul(xv2).Add(y3)
			y3 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap3, 96))).Mul(xv3).Add(y3)
		}
		if n&2 != 0 {
			pair = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(ap0, pairOffset))).Mul(xv0.GetLo()).Add(pair)
			pair = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(ap1, pairOffset))).Mul(xv1.GetLo()).Add(pair)
			pair = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(ap2, pairOffset))).Mul(xv2.GetLo()).Add(pair)
			pair = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(ap3, pairOffset))).Mul(xv3.GetLo()).Add(pair)
		}
		if n&1 != 0 {
			odd = archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(ap0, oddOffset))).AsFloat64x2().Mul(xv0.GetLo()).Add(odd)
			odd = archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(ap1, oddOffset))).AsFloat64x2().Mul(xv1.GetLo()).Add(odd)
			odd = archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(ap2, oddOffset))).AsFloat64x2().Mul(xv2.GetLo()).Add(odd)
			odd = archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(ap3, oddOffset))).AsFloat64x2().Mul(xv3.GetLo()).Add(odd)
		}
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
		if n >= 8 {
			y1 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap, 32))).Mul(xv).Add(y1)
		}
		if n >= 12 {
			y2 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap, 64))).Mul(xv).Add(y2)
		}
		if n == 16 {
			y3 = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(ap, 96))).Mul(xv).Add(y3)
		}
		if n&2 != 0 {
			pair = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(ap, pairOffset))).Mul(xv.GetLo()).Add(pair)
		}
		if n&1 != 0 {
			odd = archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(ap, oddOffset))).AsFloat64x2().Mul(xv.GetLo()).Add(odd)
		}
		m--
		if m == 0 {
			break
		}
		xp = unsafe.Add(xp, 8)
		ap = unsafe.Add(ap, rowBytes)
	}
	y0.StoreArray((*[4]float64)(yp))
	if n >= 8 {
		y1.StoreArray((*[4]float64)(unsafe.Add(yp, 32)))
	}
	if n >= 12 {
		y2.StoreArray((*[4]float64)(unsafe.Add(yp, 64)))
	}
	if n == 16 {
		y3.StoreArray((*[4]float64)(unsafe.Add(yp, 96)))
	}
	if n&2 != 0 {
		pair.StoreArray((*[2]float64)(unsafe.Add(yp, pairOffset)))
	}
	if n&1 != 0 {
		*(*uint64)(unsafe.Add(yp, oddOffset)) = odd.ToBits().GetElem(0)
	}
	archsimd.ClearAVXUpperBits()
}
