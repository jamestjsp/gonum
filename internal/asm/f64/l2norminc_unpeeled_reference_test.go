// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

func l2NormIncUnpeeledReference(x []float64, n, incX uintptr) float64 {
	base := unsafe.Pointer(unsafe.SliceData(x))
	step, offset, remaining := incX*8, uintptr(0), n
	var s0, s1, s2, s3, c0, c1, c2, c3 archsimd.Float64x4
	sign := archsimd.BroadcastUint64x4(0x8000000000000000)
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
