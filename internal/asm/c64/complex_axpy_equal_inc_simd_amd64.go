// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"simd/archsimd"
	"unsafe"
)

// The caller has validated native eligibility, complete spans, permitted aliases,
// and a common nonzero stride. Keep one byte step for all three address streams.
func complexAxpyEqualIncUncheckedSIMD(dst unsafe.Pointer, alpha complex64, x, y unsafe.Pointer, n, step uintptr) {
	ar, ai := archsimd.BroadcastFloat32x4(real(alpha)), archsimd.BroadcastFloat32x4(imag(alpha))
	// Leave at least one element after each block. Advancing the pointers then
	// stays within the validated allocation, including descending strides.
	for ; n > 4; n -= 4 {
		x0, y0 := complexPairAtSIMD(x, 0, step), complexPairAtSIMD(y, 0, step)
		x1, y1 := complexPairAtSIMD(x, 2*step, step), complexPairAtSIMD(y, 2*step, step)
		r0 := x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 0, 7, 6, x0).Mul(ai)).Add(y0).AsUint64x2()
		r1 := x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalars(1, 0, 7, 6, x1).Mul(ai)).Add(y1).AsUint64x2()
		*(*uint64)(dst) = r0.GetElem(0)
		*(*uint64)(unsafe.Add(dst, step)) = r0.GetElem(1)
		*(*uint64)(unsafe.Add(dst, 2*step)) = r1.GetElem(0)
		*(*uint64)(unsafe.Add(dst, 3*step)) = r1.GetElem(1)
		x, y, dst = unsafe.Add(x, 4*step), unsafe.Add(y, 4*step), unsafe.Add(dst, 4*step)
	}
	for ; n >= 2; n -= 2 {
		x0, y0 := complexPairAtSIMD(x, 0, step), complexPairAtSIMD(y, 0, step)
		r := x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 0, 7, 6, x0).Mul(ai)).Add(y0).AsUint64x2()
		*(*uint64)(dst) = r.GetElem(0)
		*(*uint64)(unsafe.Add(dst, step)) = r.GetElem(1)
		if n == 2 {
			return
		}
		x, y, dst = unsafe.Add(x, 2*step), unsafe.Add(y, 2*step), unsafe.Add(dst, 2*step)
	}
	if n != 0 {
		// Preserve Go's widened complex64 multiplication for the scalar
		// remainder used by the established checked pair kernel.
		*(*complex64)(dst) = alpha**(*complex64)(x) + *(*complex64)(y)
	}
}

// The caller has validated native eligibility, complete spans, permitted aliases,
// and a common nonzero stride. The destination is y; keep only two pointer streams.
func complexAxpyEqualIncInPlaceUncheckedSIMD(alpha complex64, x, y unsafe.Pointer, n, step uintptr) {
	ar, ai := archsimd.BroadcastFloat32x4(real(alpha)), archsimd.BroadcastFloat32x4(imag(alpha))
	// Leave at least one element after each block. Advancing the pointers then
	// stays within the validated allocation, including descending strides.
	for ; n > 4; n -= 4 {
		x0, y0 := complexPairAtSIMD(x, 0, step), complexPairAtSIMD(y, 0, step)
		x1, y1 := complexPairAtSIMD(x, 2*step, step), complexPairAtSIMD(y, 2*step, step)
		r0 := x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 0, 7, 6, x0).Mul(ai)).Add(y0).AsUint64x2()
		r1 := x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalars(1, 0, 7, 6, x1).Mul(ai)).Add(y1).AsUint64x2()
		*(*uint64)(y) = r0.GetElem(0)
		*(*uint64)(unsafe.Add(y, step)) = r0.GetElem(1)
		*(*uint64)(unsafe.Add(y, 2*step)) = r1.GetElem(0)
		*(*uint64)(unsafe.Add(y, 3*step)) = r1.GetElem(1)
		x, y = unsafe.Add(x, 4*step), unsafe.Add(y, 4*step)
	}
	for ; n >= 2; n -= 2 {
		x0, y0 := complexPairAtSIMD(x, 0, step), complexPairAtSIMD(y, 0, step)
		r := x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 0, 7, 6, x0).Mul(ai)).Add(y0).AsUint64x2()
		*(*uint64)(y) = r.GetElem(0)
		*(*uint64)(unsafe.Add(y, step)) = r.GetElem(1)
		if n == 2 {
			return
		}
		x, y = unsafe.Add(x, 2*step), unsafe.Add(y, 2*step)
	}
	if n != 0 {
		// Preserve Go's widened complex64 multiplication for the scalar
		// remainder used by the established checked pair kernel.
		*(*complex64)(y) = alpha**(*complex64)(x) + *(*complex64)(y)
	}
}
