// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"simd/archsimd"
	"unsafe"
)

func complexDscalShortSIMD(alpha float64, x []complex128) {
	a := archsimd.BroadcastFloat64x4(alpha)
	p := unsafe.Pointer(unsafe.SliceData(x))
	n, offset := len(x), uintptr(0)
	for ; n >= 8; n -= 8 {
		x0 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset)))
		x1 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset+32)))
		x2 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset+64)))
		x3 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset+96)))
		x0.Mul(a).StoreArray((*[4]float64)(unsafe.Add(p, offset)))
		x1.Mul(a).StoreArray((*[4]float64)(unsafe.Add(p, offset+32)))
		x2.Mul(a).StoreArray((*[4]float64)(unsafe.Add(p, offset+64)))
		x3.Mul(a).StoreArray((*[4]float64)(unsafe.Add(p, offset+96)))
		offset += 128
	}
	for ; n >= 2; n -= 2 {
		xv := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset)))
		xv.Mul(a).StoreArray((*[4]float64)(unsafe.Add(p, offset)))
		offset += 32
	}
	if n != 0 {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(p, offset)))
		xv.Mul(a.GetLo()).StoreArray((*[2]float64)(unsafe.Add(p, offset)))
	}
}

// Construct broadcasts inside the leaf: passing wide vector arguments through
// the ordinary Go ABI otherwise stages them on the stack for short calls.
func complexAxpyShortSIMD(dst, x, y []complex128, alpha complex128) {
	n := len(x)
	y, dst = y[:n:n], dst[:n:n]
	xp, yp, dp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(dst))
	ar, ai := archsimd.BroadcastFloat64x4(real(alpha)), archsimd.BroadcastFloat64x4(imag(alpha))
	var offset uintptr
	for ; n >= 4; n -= 4 {
		x0 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, offset)))
		x1 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, offset+32)))
		y0 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, offset)))
		y1 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, offset+32)))
		x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalarsGrouped(1, 2, x0).Mul(ai)).Add(y0).StoreArray((*[4]float64)(unsafe.Add(dp, offset)))
		x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalarsGrouped(1, 2, x1).Mul(ai)).Add(y1).StoreArray((*[4]float64)(unsafe.Add(dp, offset+32)))
		offset += 64
	}
	if n >= 2 {
		xv := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, offset)))
		yv := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, offset)))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalarsGrouped(1, 2, xv).Mul(ai)).Add(yv).StoreArray((*[4]float64)(unsafe.Add(dp, offset)))
		offset += 32
		n -= 2
	}
	if n != 0 {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, offset)))
		yv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, offset)))
		xv.Mul(ar.GetLo()).AddOddSubEven(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai.GetLo())).Add(yv).StoreArray((*[2]float64)(unsafe.Add(dp, offset)))
	}
}

func complexScalShortSIMD(alpha complex128, x []complex128) {
	ar, ai := archsimd.BroadcastFloat64x4(real(alpha)), archsimd.BroadcastFloat64x4(imag(alpha))
	p := unsafe.Pointer(unsafe.SliceData(x))
	n, offset := len(x), uintptr(0)
	for ; n >= 4; n -= 4 {
		x0 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset)))
		x1 := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset+32)))
		x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalarsGrouped(1, 2, x0).Mul(ai)).StoreArray((*[4]float64)(unsafe.Add(p, offset)))
		x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalarsGrouped(1, 2, x1).Mul(ai)).StoreArray((*[4]float64)(unsafe.Add(p, offset+32)))
		offset += 64
	}
	if n >= 2 {
		xv := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(p, offset)))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalarsGrouped(1, 2, xv).Mul(ai)).StoreArray((*[4]float64)(unsafe.Add(p, offset)))
		offset += 32
		n -= 2
	}
	if n != 0 {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(p, offset)))
		xv.Mul(ar.GetLo()).AddOddSubEven(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai.GetLo())).StoreArray((*[2]float64)(unsafe.Add(p, offset)))
	}
}
