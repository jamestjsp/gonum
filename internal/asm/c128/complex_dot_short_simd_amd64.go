// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"simd/archsimd"
	"unsafe"
)

// Short contiguous dots avoid the wide loop's component scratch and scalar
// complex tail. The caller preserves the complete original exceptional retry.
func complexDotShortSIMD(x, y []complex128, conjugate bool) complex128 {
	y = y[:len(x):len(x)]
	xp, yp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y))
	n, offset := len(x), uintptr(0)
	var r0, r1, i0, i1 archsimd.Float64x4
	for ; n >= 4; n -= 4 {
		x0, y0 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, offset))), archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, offset)))
		x1, y1 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, offset+32))), archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, offset+32)))
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalarsGrouped(1, 2, y0)))
		r1, i1 = r1.Add(x1.Mul(y1)), i1.Add(x1.Mul(y1.ConcatPermuteScalarsGrouped(1, 2, y1)))
		offset += 64
	}
	r0, i0 = r0.Add(r1), i0.Add(i1)
	if n >= 2 {
		xv, yv := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, offset))), archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, offset)))
		r0, i0 = r0.Add(xv.Mul(yv)), i0.Add(xv.Mul(yv.ConcatPermuteScalarsGrouped(1, 2, yv)))
		offset += 32
		n -= 2
	}
	r, im := r0.GetLo().Add(r0.GetHi()), i0.GetLo().Add(i0.GetHi())
	if n != 0 {
		xv, yv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, offset))), archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, offset)))
		r, im = r.Add(xv.Mul(yv)), im.Add(xv.Mul(yv.ConcatPermuteScalars(1, 2, yv)))
	}
	var sign archsimd.Uint64x2
	sign = sign.SetElem(1, 1<<63)
	if conjugate {
		im = im.ToBits().Xor(sign).BitsToFloat64()
	} else {
		r = r.ToBits().Xor(sign).BitsToFloat64()
	}
	result := r.ConcatAddPairs(im)
	return complex(result.GetElem(0), result.GetElem(1))
}
