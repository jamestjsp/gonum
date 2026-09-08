// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"simd/archsimd"
	"unsafe"
)

// Component accumulation replaces a horizontal add for every pair with two
// reductions after the loop. The caller retains the original short grouping
// as a cold fallback because component accumulators can overflow sooner.
func complexDotShortFastSIMD(x, y []complex64, conjugate bool) complex64 {
	y = y[:len(x):len(x)]
	r0, r1 := archsimd.BroadcastFloat32x4(0), archsimd.BroadcastFloat32x4(0)
	i0, i1 := archsimd.BroadcastFloat32x4(0), archsimd.BroadcastFloat32x4(0)
	for len(x) >= 4 {
		x0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[0])))
		y0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[0])))
		x1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[2])))
		y1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[2])))
		r0 = r0.Add(x0.Mul(y0))
		i0 = i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0)))
		r1 = r1.Add(x1.Mul(y1))
		i1 = i1.Add(x1.Mul(y1.ConcatPermuteScalars(1, 0, 7, 6, y1)))
		x, y = x[4:], y[4:]
	}
	r0, i0 = r0.Add(r1), i0.Add(i1)
	if len(x) >= 2 {
		xv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[0])))
		yv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[0])))
		r0 = r0.Add(xv.Mul(yv))
		i0 = i0.Add(xv.Mul(yv.ConcatPermuteScalars(1, 0, 7, 6, yv)))
		x, y = x[2:], y[2:]
	}
	sign := archsimd.BroadcastUint64x2(1 << 63).AsUint32x4()
	if conjugate {
		i0 = i0.ToBits().Xor(sign).BitsToFloat32()
	} else {
		r0 = r0.ToBits().Xor(sign).BitsToFloat32()
	}
	result := r0.ConcatAddPairs(i0)
	result = result.ConcatAddPairs(result)
	sum := complex(result.GetElem(0), result.GetElem(1))
	if len(x) != 0 {
		value := x[0]
		if conjugate {
			value = conj64(value)
		}
		sum += value * y[0]
	}
	return sum
}

// The wide short loop loads four complex values at a time. Its caller keeps
// both established short groupings when component accumulation is nonfinite.
func complexDotShortWideSIMD(x, y []complex64, conjugate bool) complex64 {
	y = y[:len(x):len(x)]
	xp, yp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y))
	n, offset := len(x), uintptr(0)
	var r0, r1, i0, i1 archsimd.Float32x8
	for ; n >= 8; n -= 8 {
		x0, y0 := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, offset))), archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, offset)))
		x1, y1 := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, offset+32))), archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, offset+32)))
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalarsGrouped(1, 0, 7, 6, y0)))
		r1, i1 = r1.Add(x1.Mul(y1)), i1.Add(x1.Mul(y1.ConcatPermuteScalarsGrouped(1, 0, 7, 6, y1)))
		offset += 64
	}
	r0, i0 = r0.Add(r1), i0.Add(i1)
	if n >= 4 {
		xv, yv := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, offset))), archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, offset)))
		r0, i0 = r0.Add(xv.Mul(yv)), i0.Add(xv.Mul(yv.ConcatPermuteScalarsGrouped(1, 0, 7, 6, yv)))
		offset += 32
		n -= 4
	}
	r, im := r0.GetLo().Add(r0.GetHi()), i0.GetLo().Add(i0.GetHi())
	if n >= 2 {
		xv, yv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, offset))), archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(yp, offset)))
		r, im = r.Add(xv.Mul(yv)), im.Add(xv.Mul(yv.ConcatPermuteScalars(1, 0, 7, 6, yv)))
		offset += 16
		n -= 2
	}
	if n != 0 {
		var xv, yv archsimd.Uint64x2
		x0 := xv.SetElem(0, *(*uint64)(unsafe.Add(xp, offset))).AsFloat32x4()
		y0 := yv.SetElem(0, *(*uint64)(unsafe.Add(yp, offset))).AsFloat32x4()
		r, im = r.Add(x0.Mul(y0)), im.Add(x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0)))
	}
	sign := archsimd.BroadcastUint64x2(1 << 63).AsUint32x4()
	if conjugate {
		im = im.ToBits().Xor(sign).BitsToFloat32()
	} else {
		r = r.ToBits().Xor(sign).BitsToFloat32()
	}
	result := r.ConcatAddPairs(im)
	result = result.ConcatAddPairs(result)
	return complex(result.GetElem(0), result.GetElem(1))
}
