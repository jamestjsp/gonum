// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

func complexStrideInBoundsSIMD(length int, n, inc, index uintptr) bool {
	if n == 0 {
		return true
	}
	if index >= uintptr(length) {
		return false
	}
	if int(inc) < 0 {
		hi, lo := bits.Mul64(uint64(n-1), uint64(-inc))
		return hi == 0 && lo <= uint64(index)
	}
	hi, lo := bits.Mul64(uint64(n-1), uint64(inc))
	return hi == 0 && lo < uint64(uintptr(length)-index)
}

func complexDotIncNativeSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr, conjugate bool) complex64 {
	if n == 0 {
		return 0
	}
	if complexStrideInBoundsSIMD(len(x), n, incX, ix) && complexStrideInBoundsSIMD(len(y), n, incY, iy) {
		var sum complex64
		if n >= 64 {
			sum = complexDotIncEightSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8, conjugate)
		} else {
			sum = complexDotIncUncheckedSIMD(unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*8, incY*8, conjugate)
		}
		if math.Float32bits(real(sum))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(sum))&0x7f800000 != 0x7f800000 {
			return sum
		}
	}
	// Component sums can overflow sooner than per-complex products. Preserve
	// the complete original kernel, including its portable/sequential recovery.
	return complexDotIncCheckedSIMD(x, y, n, incX, incY, ix, iy, conjugate)
}

func complexPairAtSIMD(base unsafe.Pointer, offset, step uintptr) archsimd.Float32x4 {
	var pair archsimd.Uint64x2
	pair = pair.SetElem(0, *(*uint64)(unsafe.Add(base, offset)))
	pair = pair.SetElem(1, *(*uint64)(unsafe.Add(base, offset+step)))
	return pair.AsFloat32x4()
}

// All addresses and permitted aliases have been validated by the caller.
// Pointer advances leave at least one accessible element for the next access.
func complexAxpyIncUncheckedSIMD(dst unsafe.Pointer, sd uintptr, alpha complex64, x, y unsafe.Pointer, n, sx, sy uintptr) {
	ar, ai := archsimd.BroadcastFloat32x4(real(alpha)), archsimd.BroadcastFloat32x4(imag(alpha))
	// Leave at least one element after each block. Advancing the pointers then
	// stays within the validated allocation, including descending strides.
	for ; n > 4; n -= 4 {
		x0, y0 := complexPairAtSIMD(x, 0, sx), complexPairAtSIMD(y, 0, sy)
		x1, y1 := complexPairAtSIMD(x, 2*sx, sx), complexPairAtSIMD(y, 2*sy, sy)
		r0 := x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 0, 7, 6, x0).Mul(ai)).Add(y0).AsUint64x2()
		r1 := x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalars(1, 0, 7, 6, x1).Mul(ai)).Add(y1).AsUint64x2()
		*(*uint64)(dst) = r0.GetElem(0)
		*(*uint64)(unsafe.Add(dst, sd)) = r0.GetElem(1)
		*(*uint64)(unsafe.Add(dst, 2*sd)) = r1.GetElem(0)
		*(*uint64)(unsafe.Add(dst, 3*sd)) = r1.GetElem(1)
		x, y, dst = unsafe.Add(x, 4*sx), unsafe.Add(y, 4*sy), unsafe.Add(dst, 4*sd)
	}
	for ; n >= 2; n -= 2 {
		x0, y0 := complexPairAtSIMD(x, 0, sx), complexPairAtSIMD(y, 0, sy)
		r := x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 0, 7, 6, x0).Mul(ai)).Add(y0).AsUint64x2()
		*(*uint64)(dst) = r.GetElem(0)
		*(*uint64)(unsafe.Add(dst, sd)) = r.GetElem(1)
		if n == 2 {
			return
		}
		x, y, dst = unsafe.Add(x, 2*sx), unsafe.Add(y, 2*sy), unsafe.Add(dst, 2*sd)
	}
	if n != 0 {
		// Preserve Go's widened complex64 multiplication for the scalar
		// remainder used by the established checked pair kernel.
		*(*complex64)(dst) = alpha**(*complex64)(x) + *(*complex64)(y)
	}
}

func complexDotIncUncheckedSIMD(x, y unsafe.Pointer, n, sx, sy uintptr, conjugate bool) complex64 {
	var r0, r1, i0, i1 archsimd.Float32x4
	var ix, iy uintptr
	for ; n >= 4; n -= 4 {
		x0, y0 := complexPairAtSIMD(x, ix, sx), complexPairAtSIMD(y, iy, sy)
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0)))
		x1, y1 := complexPairAtSIMD(x, ix+2*sx, sx), complexPairAtSIMD(y, iy+2*sy, sy)
		r1, i1 = r1.Add(x1.Mul(y1)), i1.Add(x1.Mul(y1.ConcatPermuteScalars(1, 0, 7, 6, y1)))
		ix, iy = ix+4*sx, iy+4*sy
	}
	r0, i0 = r0.Add(r1), i0.Add(i1)
	for ; n >= 2; n -= 2 {
		xv, yv := complexPairAtSIMD(x, ix, sx), complexPairAtSIMD(y, iy, sy)
		r0, i0 = r0.Add(xv.Mul(yv)), i0.Add(xv.Mul(yv.ConcatPermuteScalars(1, 0, 7, 6, yv)))
		ix, iy = ix+2*sx, iy+2*sy
	}
	if n != 0 {
		var xv, yv archsimd.Uint64x2
		x0 := xv.SetElem(0, *(*uint64)(unsafe.Add(x, ix))).AsFloat32x4()
		y0 := yv.SetElem(0, *(*uint64)(unsafe.Add(y, iy))).AsFloat32x4()
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0)))
	}
	sign := archsimd.BroadcastUint64x2(1 << 63).AsUint32x4()
	if conjugate {
		i0 = i0.ToBits().Xor(sign).BitsToFloat32()
	} else {
		r0 = r0.ToBits().Xor(sign).BitsToFloat32()
	}
	result := r0.ConcatAddPairs(i0)
	result = result.ConcatAddPairs(result)
	return complex(result.GetElem(0), result.GetElem(1))
}

func complexDotIncEightSIMD(x, y unsafe.Pointer, n, sx, sy uintptr, conjugate bool) complex64 {
	var r0, r1, r2, r3, i0, i1, i2, i3 archsimd.Float32x4
	var ix, iy uintptr
	for ; n >= 8; n -= 8 {
		x0, y0 := complexPairAtSIMD(x, ix, sx), complexPairAtSIMD(y, iy, sy)
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0)))
		x1, y1 := complexPairAtSIMD(x, ix+2*sx, sx), complexPairAtSIMD(y, iy+2*sy, sy)
		r1, i1 = r1.Add(x1.Mul(y1)), i1.Add(x1.Mul(y1.ConcatPermuteScalars(1, 0, 7, 6, y1)))
		x2, y2 := complexPairAtSIMD(x, ix+4*sx, sx), complexPairAtSIMD(y, iy+4*sy, sy)
		r2, i2 = r2.Add(x2.Mul(y2)), i2.Add(x2.Mul(y2.ConcatPermuteScalars(1, 0, 7, 6, y2)))
		x3, y3 := complexPairAtSIMD(x, ix+6*sx, sx), complexPairAtSIMD(y, iy+6*sy, sy)
		r3, i3 = r3.Add(x3.Mul(y3)), i3.Add(x3.Mul(y3.ConcatPermuteScalars(1, 0, 7, 6, y3)))
		ix, iy = ix+8*sx, iy+8*sy
	}
	r0, i0 = r0.Add(r1).Add(r2.Add(r3)), i0.Add(i1).Add(i2.Add(i3))
	for ; n >= 2; n -= 2 {
		xv, yv := complexPairAtSIMD(x, ix, sx), complexPairAtSIMD(y, iy, sy)
		r0, i0 = r0.Add(xv.Mul(yv)), i0.Add(xv.Mul(yv.ConcatPermuteScalars(1, 0, 7, 6, yv)))
		ix, iy = ix+2*sx, iy+2*sy
	}
	if n != 0 {
		var xv, yv archsimd.Uint64x2
		x0 := xv.SetElem(0, *(*uint64)(unsafe.Add(x, ix))).AsFloat32x4()
		y0 := yv.SetElem(0, *(*uint64)(unsafe.Add(y, iy))).AsFloat32x4()
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0)))
	}
	sign := archsimd.BroadcastUint64x2(1 << 63).AsUint32x4()
	if conjugate {
		i0 = i0.ToBits().Xor(sign).BitsToFloat32()
	} else {
		r0 = r0.ToBits().Xor(sign).BitsToFloat32()
	}
	result := r0.ConcatAddPairs(i0)
	result = result.ConcatAddPairs(result)
	return complex(result.GetElem(0), result.GetElem(1))
}
