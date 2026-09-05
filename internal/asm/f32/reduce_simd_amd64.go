// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"simd"
	"simd/archsimd"
)

// The portable API lacks horizontal addition. Reduce once after the vector
// loop, keeping the lane tree in registers instead of spilling every lane.
func reduceF32(value simd.Float32s) float32 {
	if simd.Emulated() {
		return reduceF32Portable(value)
	}
	var quarter archsimd.Float32x4
	switch v := value.ToArch().(type) {
	case archsimd.Float32x16:
		half := v.GetLo().Add(v.GetHi())
		quarter = half.GetLo().Add(half.GetHi())
	case archsimd.Float32x8:
		quarter = v.GetLo().Add(v.GetHi())
	case archsimd.Float32x4:
		quarter = v
	default:
		return reduceF32Portable(value)
	}
	pair := quarter.Add(quarter.ConcatPermuteScalars(2, 3, 0, 1, quarter))
	sum := pair.GetElem(0) + pair.GetElem(1)
	if math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
		// Preserve the original lane order before the caller considers a
		// whole-input retry: that retry can overflow even when these lanes
		// sum to a finite result in their original order.
		return reduceF32Portable(value)
	}
	return sum
}
