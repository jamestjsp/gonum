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

// Two exact scalar loads form each pair; increment gaps are never read.
func gatherNormPairSIMD(base unsafe.Pointer, offset, step uintptr) archsimd.Float64x2 {
	var v archsimd.Uint64x2
	v = v.SetElem(0, *(*uint64)(unsafe.Add(base, offset)))
	v = v.SetElem(1, *(*uint64)(unsafe.Add(base, offset+step)))
	return v.AsFloat64x2()
}

func l1NormIncHardwareSIMD(x []float64, n, inc int) (float64, bool) {
	if n < 32 || !archsimd.X86.AVX2() || simd.Emulated() || !simdPositiveSpan(len(x), uintptr(n), uintptr(inc), 0) {
		return 0, false
	}
	base := unsafe.Pointer(unsafe.SliceData(x))
	step, offset := uintptr(inc)*8, uintptr(0)
	var s0, s1, s2, s3 archsimd.Float64x2
	absMask := archsimd.BroadcastUint64x2(1<<63 - 1)
	for ; n >= 8; n -= 8 {
		block := unsafe.Add(base, offset)
		s0 = s0.Add(gatherNormPairSIMD(block, 0, step).AsUint64x2().And(absMask).AsFloat64x2())
		s1 = s1.Add(gatherNormPairSIMD(block, 2*step, step).AsUint64x2().And(absMask).AsFloat64x2())
		s2 = s2.Add(gatherNormPairSIMD(block, 4*step, step).AsUint64x2().And(absMask).AsFloat64x2())
		s3 = s3.Add(gatherNormPairSIMD(block, 6*step, step).AsUint64x2().And(absMask).AsFloat64x2())
		offset += 8 * step
	}
	s0 = s0.Add(s1).Add(s2.Add(s3))
	sum := s0.GetElem(0) + s0.GetElem(1)
	for ; n > 0; n-- {
		sum += math.Abs(*simdStrideAt(base, offset))
		offset += step
	}
	return sum, true
}
