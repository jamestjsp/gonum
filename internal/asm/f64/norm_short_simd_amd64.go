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

func normShortHardwareAvailableSIMD() bool {
	return !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() && archsimd.X86.FMA()
}

// The final tree reduction carries the rounding error at every addition.
func normFinishNative256(sum, correction archsimd.Float64x4, n int) float64 {
	a, b := sum.GetLo(), sum.GetHi()
	next := a.Add(b)
	virtual := next.Sub(a)
	err := a.Sub(next.Sub(virtual)).Add(b.Sub(virtual))
	corr := correction.GetLo().Add(correction.GetHi()).Add(err)
	var reduced [4]float64
	next.StoreArray((*[2]float64)(reduced[:2]))
	corr.StoreArray((*[2]float64)(reduced[2:]))
	archsimd.ClearAVXUpperBits()
	return normFinishShortScalar(&reduced, n)

}

//go:noinline
func normFinishShortScalar(v *[4]float64, n int) float64 {
	s, e := normTwoSumScalar(v[0], v[1])
	sum := s + (e + v[2] + v[3])
	if normSumUsable(sum, n) {
		return math.Sqrt(sum)
	}
	return math.NaN()
}

func loadNormTailNative256(x []float64) archsimd.Float64x4 {
	var lo archsimd.Uint64x2
	hi := archsimd.BroadcastUint64x2(0)
	lo = lo.SetElem(0, *(*uint64)(unsafe.Pointer(&x[0])))
	if len(x) > 1 {
		lo = lo.SetElem(1, *(*uint64)(unsafe.Pointer(&x[1])))
	}
	if len(x) > 2 {
		hi = hi.SetElem(0, *(*uint64)(unsafe.Pointer(&x[2])))
	}
	var v archsimd.Uint64x4
	return v.SetLo(lo).SetHi(hi).AsFloat64x4()
}

func l2NormShortNativeSIMD(x []float64) float64 {
	var s0, s1, s2, s3, c0, c1, c2, c3 archsimd.Float64x4
	sign := archsimd.BroadcastUint64x4(0x8000000000000000)
	i := 0
	for ; i+16 <= len(x); i += 16 {
		s0, c0 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+0:i+4])), s0, c0, sign)
		s1, c1 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+4:i+8])), s1, c1, sign)
		s2, c2 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+8:i+12])), s2, c2, sign)
		s3, c3 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+12:i+16])), s3, c3, sign)
	}
	if len(x) >= 16 {
		s0, c0 = normMergeNative256(s0, c0, s1, c1)
		s2, c2 = normMergeNative256(s2, c2, s3, c3)
		s0, c0 = normMergeNative256(s0, c0, s2, c2)
	}
	for ; i+4 <= len(x); i += 4 {
		s0, c0 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i:i+4])), s0, c0, sign)
	}
	if i < len(x) {
		s0, c0 = normSquareNative256(loadNormTailNative256(x[i:]), s0, c0, sign)
	}
	norm := normFinishNative256(s0, c0, len(x))
	if math.Float64bits(norm)&0x7ff0000000000000 != 0x7ff0000000000000 {
		return norm
	}
	return l2NormUnitaryScalar(x)
}

func l2DistanceShortNativeSIMD(x, y []float64) float64 {
	y = y[:len(x):len(x)]
	var s0, s1, s2, s3, c0, c1, c2, c3 archsimd.Float64x4
	sign := archsimd.BroadcastUint64x4(0x8000000000000000)
	i := 0
	for ; i+16 <= len(x); i += 16 {
		s0, c0 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+0:i+4])).Sub(archsimd.LoadFloat64x4Array((*[4]float64)(y[i+0:i+4]))), s0, c0, sign)
		s1, c1 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+4:i+8])).Sub(archsimd.LoadFloat64x4Array((*[4]float64)(y[i+4:i+8]))), s1, c1, sign)
		s2, c2 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+8:i+12])).Sub(archsimd.LoadFloat64x4Array((*[4]float64)(y[i+8:i+12]))), s2, c2, sign)
		s3, c3 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i+12:i+16])).Sub(archsimd.LoadFloat64x4Array((*[4]float64)(y[i+12:i+16]))), s3, c3, sign)
	}
	if len(x) >= 16 {
		s0, c0 = normMergeNative256(s0, c0, s1, c1)
		s2, c2 = normMergeNative256(s2, c2, s3, c3)
		s0, c0 = normMergeNative256(s0, c0, s2, c2)
	}
	for ; i+4 <= len(x); i += 4 {
		s0, c0 = normSquareNative256(archsimd.LoadFloat64x4Array((*[4]float64)(x[i:i+4])).Sub(archsimd.LoadFloat64x4Array((*[4]float64)(y[i:i+4]))), s0, c0, sign)
	}
	if i < len(x) {
		s0, c0 = normSquareNative256(loadNormTailNative256(x[i:]).Sub(loadNormTailNative256(y[i:])), s0, c0, sign)
	}
	norm := normFinishNative256(s0, c0, len(x))
	if math.Float64bits(norm)&0x7ff0000000000000 != 0x7ff0000000000000 {
		return norm
	}
	return l2DistanceUnitaryScalar(x, y)
}
