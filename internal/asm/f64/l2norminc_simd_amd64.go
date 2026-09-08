// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

func l2NormIncHardwareSIMD(x []float64, n, incX uintptr) (float64, bool) {
	if n < 16 || incX < 2 || simd.Emulated() || simd.BroadcastFloat64s(0).Len() < 4 || !archsimd.X86.AVX2() || !archsimd.X86.FMA() || !simdPositiveSpan(len(x), n, incX, 0) {
		return 0, false
	}
	return l2NormIncNative256(x, n, incX), true
}

// Read exactly four selected elements. Loading the contiguous span would also
// read inactive gaps, which may belong to a concurrent caller.
func gatherNormQuadSIMD(base unsafe.Pointer, offset, step uintptr) archsimd.Float64x4 {
	var lo, hi archsimd.Uint64x2
	lo = lo.SetElem(0, *(*uint64)(unsafe.Add(base, offset)))
	lo = lo.SetElem(1, *(*uint64)(unsafe.Add(base, offset+step)))
	hi = hi.SetElem(0, *(*uint64)(unsafe.Add(base, offset+2*step)))
	hi = hi.SetElem(1, *(*uint64)(unsafe.Add(base, offset+3*step)))
	var v archsimd.Uint64x4
	return v.SetLo(lo).SetHi(hi).AsFloat64x4()
}

// Keep the existing TwoSum/TwoProductFMA compensation, with four independent
// chains and register gathering instead of the portable lane scratch loop.
func l2NormIncNative256(x []float64, n, incX uintptr) float64 {
	base := unsafe.Pointer(unsafe.SliceData(x))
	step, offset, remaining := incX*8, uintptr(0), n
	sign := archsimd.BroadcastUint64x4(0x8000000000000000)
	// Admission guarantees at least one full block. Seed its four chains
	// without TwoSum(+0, product), preserving positive-zero corrections.
	s0, c0 := normSeedNative256(gatherNormQuadSIMD(base, offset, step), sign)
	s1, c1 := normSeedNative256(gatherNormQuadSIMD(base, offset+4*step, step), sign)
	s2, c2 := normSeedNative256(gatherNormQuadSIMD(base, offset+8*step, step), sign)
	s3, c3 := normSeedNative256(gatherNormQuadSIMD(base, offset+12*step, step), sign)
	offset += 16 * step
	remaining -= 16
	for ; remaining >= 16; remaining -= 16 {
		s0, c0 = normSquareNative256(gatherNormQuadSIMD(base, offset, step), s0, c0, sign)
		s1, c1 = normSquareNative256(gatherNormQuadSIMD(base, offset+4*step, step), s1, c1, sign)
		s2, c2 = normSquareNative256(gatherNormQuadSIMD(base, offset+8*step, step), s2, c2, sign)
		s3, c3 = normSquareNative256(gatherNormQuadSIMD(base, offset+12*step, step), s3, c3, sign)
		offset += 16 * step
	}
	s0, c0 = normMergeNative256(s0, c0, s1, c1)
	s0, c0 = normMergeNative256(s0, c0, s2, c2)
	s0, c0 = normMergeNative256(s0, c0, s3, c3)
	for ; remaining >= 4; remaining -= 4 {
		s0, c0 = normSquareNative256(gatherNormQuadSIMD(base, offset, step), s0, c0, sign)
		offset += 4 * step
	}
	var sums, corrections [4]float64
	s0.StoreArray(&sums)
	c0.StoreArray(&corrections)
	// The remaining reduction and scaled fallback use scalar SSE. All wide
	// values are stored, so their upper halves are now dead.
	archsimd.ClearAVXUpperBits()
	var sum, correction float64
	for i, v := range sums {
		var sumError float64
		sum, sumError = normTwoSumScalar(sum, v)
		correction += sumError + corrections[i]
	}
	for ; remaining > 0; remaining-- {
		sum, correction = normSquareScalar(*simdStrideAt(base, offset), sum, correction)
		offset += step
	}
	sum += correction
	if normSumUsable(sum, int(n)) {
		return math.Sqrt(sum)
	}
	return l2NormIncScalar(x, n, incX)
}

func normSquareNative256(v, sum, correction archsimd.Float64x4, sign archsimd.Uint64x4) (archsimd.Float64x4, archsimd.Float64x4) {
	product := v.Mul(v)
	productError := v.MulAdd(v, product.ToBits().Xor(sign).AsFloat64x4())
	next, sumError := normTwoSumNative256(sum, product)
	return next, correction.Add(sumError.Add(productError))
}

func normTwoSumNative256(a, b archsimd.Float64x4) (archsimd.Float64x4, archsimd.Float64x4) {
	sum := a.Add(b)
	bVirtual := sum.Sub(a)
	return sum, a.Sub(sum.Sub(bVirtual)).Add(b.Sub(bVirtual))
}

func normMergeNative256(sum, correction, other, otherCorrection archsimd.Float64x4) (archsimd.Float64x4, archsimd.Float64x4) {
	next, sumError := normTwoSumNative256(sum, other)
	return next, correction.Add(otherCorrection.Add(sumError))
}

// For a finite product, TwoSum(+0, product) is (product, +0). Retain one
// positive-zero addition so a rounded negative-zero FMA residual is normalized
// exactly as in the original compensated update. Nonfinite products still
// reach the unchanged scaled fallback through the mandatory chain merges.
func normSeedNative256(v archsimd.Float64x4, sign archsimd.Uint64x4) (sum, correction archsimd.Float64x4) {
	product := v.Mul(v)
	productError := v.MulAdd(v, product.ToBits().Xor(sign).AsFloat64x4())
	var zero archsimd.Float64x4
	return product, zero.Add(productError)
}
