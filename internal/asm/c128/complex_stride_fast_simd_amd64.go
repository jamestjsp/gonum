// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// Check the entire index stream before using exact-element pointer loads. The
// multiplication must not wrap: negative BLAS increments are encoded in uintptr.
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

func complexAxpyIncNativeSIMD(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incDst != 0 && complexStrideInBoundsSIMD(len(dst), n, incDst, idst) && complexStrideInBoundsSIMD(len(x), n, incX, ix) && complexStrideInBoundsSIMD(len(y), n, incY, iy) {
		complexAxpyIncUncheckedSIMD(unsafe.Pointer(&dst[idst]), incDst*16, alpha, unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), n, incX*16, incY*16)
		return
	}
	// Preserve ordered writes and bounds panics for an invalid index stream.
	complexAxpyIncCheckedSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy)
}

// The caller validates the spans and aliasing. Pointer advances leave at least
// one accessible element, including when traversing a descending index stream.
func complexAxpyIncUncheckedSIMD(dst unsafe.Pointer, incDst uintptr, alpha complex128, x, y unsafe.Pointer, n, incX, incY uintptr) {
	ar, ai := archsimd.BroadcastFloat64x2(real(alpha)), archsimd.BroadcastFloat64x2(imag(alpha))
	for ; n > 4; n -= 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		y0 := archsimd.LoadFloat64x2Array((*[2]float64)(y))
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, incX)))
		y1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, incY)))
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 2*incX)))
		y2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, 2*incY)))
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 3*incX)))
		y3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, 3*incY)))
		x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 2, x0).Mul(ai)).Add(y0).StoreArray((*[2]float64)(dst))
		x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalars(1, 2, x1).Mul(ai)).Add(y1).StoreArray((*[2]float64)(unsafe.Add(dst, incDst)))
		x2.Mul(ar).AddOddSubEven(x2.ConcatPermuteScalars(1, 2, x2).Mul(ai)).Add(y2).StoreArray((*[2]float64)(unsafe.Add(dst, 2*incDst)))
		x3.Mul(ar).AddOddSubEven(x3.ConcatPermuteScalars(1, 2, x3).Mul(ai)).Add(y3).StoreArray((*[2]float64)(unsafe.Add(dst, 3*incDst)))
		x, y, dst = unsafe.Add(x, 4*incX), unsafe.Add(y, 4*incY), unsafe.Add(dst, 4*incDst)
	}
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		yv := archsimd.LoadFloat64x2Array((*[2]float64)(y))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai)).Add(yv).StoreArray((*[2]float64)(dst))
		if n == 1 {
			return
		}
		x, y, dst = unsafe.Add(x, incX), unsafe.Add(y, incY), unsafe.Add(dst, incDst)
	}
}

// Accumulate component products without a horizontal operation per element.
// The caller retries the complete established kernel if reassociation overflows.
func complexDotIncUncheckedSIMD(x, y unsafe.Pointer, n, incX, incY uintptr, conjugate bool) complex128 {
	r0, r1 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	r2, r3 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	i0, i1 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	i2, i3 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	var ix, iy uintptr
	for ; n >= 4; n -= 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, ix)))
		y0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, iy)))
		r0 = r0.Add(x0.Mul(y0))
		i0 = i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 2, y0)))
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, ix+incX)))
		y1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, iy+incY)))
		r1 = r1.Add(x1.Mul(y1))
		i1 = i1.Add(x1.Mul(y1.ConcatPermuteScalars(1, 2, y1)))
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, ix+2*incX)))
		y2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, iy+2*incY)))
		r2 = r2.Add(x2.Mul(y2))
		i2 = i2.Add(x2.Mul(y2.ConcatPermuteScalars(1, 2, y2)))
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, ix+3*incX)))
		y3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, iy+3*incY)))
		r3 = r3.Add(x3.Mul(y3))
		i3 = i3.Add(x3.Mul(y3.ConcatPermuteScalars(1, 2, y3)))
		ix += 4 * incX
		iy += 4 * incY
	}
	r0, i0 = r0.Add(r1).Add(r2).Add(r3), i0.Add(i1).Add(i2).Add(i3)
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, ix)))
		yv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, iy)))
		r0 = r0.Add(xv.Mul(yv))
		i0 = i0.Add(xv.Mul(yv.ConcatPermuteScalars(1, 2, yv)))
		ix += incX
		iy += incY
	}
	sign := archsimd.BroadcastUint64x2(0).SetElem(1, 1<<63)
	if conjugate {
		i0 = i0.ToBits().Xor(sign).BitsToFloat64()
	} else {
		r0 = r0.ToBits().Xor(sign).BitsToFloat64()
	}
	result := r0.ConcatAddPairs(i0)
	return complex(result.GetElem(0), result.GetElem(1))
}

func complexScalIncUncheckedSIMD(alpha complex128, x unsafe.Pointer, n, inc uintptr) {
	ar, ai := archsimd.BroadcastFloat64x2(real(alpha)), archsimd.BroadcastFloat64x2(imag(alpha))
	// Each advance leaves a valid next element; final blocks never advance.
	for ; n > 4; n -= 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, inc)))
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 3*inc)))
		x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 2, x0).Mul(ai)).StoreArray((*[2]float64)(x))
		x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalars(1, 2, x1).Mul(ai)).StoreArray((*[2]float64)(unsafe.Add(x, inc)))
		x2.Mul(ar).AddOddSubEven(x2.ConcatPermuteScalars(1, 2, x2).Mul(ai)).StoreArray((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3.Mul(ar).AddOddSubEven(x3.ConcatPermuteScalars(1, 2, x3).Mul(ai)).StoreArray((*[2]float64)(unsafe.Add(x, 3*inc)))
		x = unsafe.Add(x, 4*inc)
	}
	if n == 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, inc)))
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 3*inc)))
		x0.Mul(ar).AddOddSubEven(x0.ConcatPermuteScalars(1, 2, x0).Mul(ai)).StoreArray((*[2]float64)(x))
		x1.Mul(ar).AddOddSubEven(x1.ConcatPermuteScalars(1, 2, x1).Mul(ai)).StoreArray((*[2]float64)(unsafe.Add(x, inc)))
		x2.Mul(ar).AddOddSubEven(x2.ConcatPermuteScalars(1, 2, x2).Mul(ai)).StoreArray((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3.Mul(ar).AddOddSubEven(x3.ConcatPermuteScalars(1, 2, x3).Mul(ai)).StoreArray((*[2]float64)(unsafe.Add(x, 3*inc)))
		return
	}
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai)).StoreArray((*[2]float64)(x))
		if n == 1 {
			return
		}
		x = unsafe.Add(x, inc)
	}
}

func complexDscalIncUncheckedSIMD(alpha float64, x unsafe.Pointer, n, inc uintptr) {
	a := archsimd.BroadcastFloat64x2(alpha)
	// Each advance leaves a valid next element; final blocks never advance.
	for ; n > 4; n -= 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, inc)))
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 3*inc)))
		x0.Mul(a).StoreArray((*[2]float64)(x))
		x1.Mul(a).StoreArray((*[2]float64)(unsafe.Add(x, inc)))
		x2.Mul(a).StoreArray((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3.Mul(a).StoreArray((*[2]float64)(unsafe.Add(x, 3*inc)))
		x = unsafe.Add(x, 4*inc)
	}
	if n == 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, inc)))
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, 3*inc)))
		x0.Mul(a).StoreArray((*[2]float64)(x))
		x1.Mul(a).StoreArray((*[2]float64)(unsafe.Add(x, inc)))
		x2.Mul(a).StoreArray((*[2]float64)(unsafe.Add(x, 2*inc)))
		x3.Mul(a).StoreArray((*[2]float64)(unsafe.Add(x, 3*inc)))
		return
	}
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		xv.Mul(a).StoreArray((*[2]float64)(x))
		if n == 1 {
			return
		}
		x = unsafe.Add(x, inc)
	}
}

// Gather two complete complex values with exact128-bit loads; gaps are not read.
func complexPairAtSIMD(base unsafe.Pointer, offset, step uintptr) archsimd.Float64x4 {
	lo := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(base, offset)))
	hi := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(base, offset+step)))
	var pair archsimd.Float64x4
	return pair.SetLo(lo).SetHi(hi)
}

func complexDotIncPairSIMD(x, y unsafe.Pointer, n, sx, sy uintptr, conjugate bool) complex128 {
	var r0, r1, i0, i1 archsimd.Float64x4
	// Leave a valid next element before advancing either pointer, including
	// descending and zero strides. Preserve the two-pair accumulator groups.
	for ; n > 4; n -= 4 {
		x0, y0 := complexPairAtSIMD(x, 0, sx), complexPairAtSIMD(y, 0, sy)
		x1, y1 := complexPairAtSIMD(x, 2*sx, sx), complexPairAtSIMD(y, 2*sy, sy)
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalarsGrouped(1, 2, y0)))
		r1, i1 = r1.Add(x1.Mul(y1)), i1.Add(x1.Mul(y1.ConcatPermuteScalarsGrouped(1, 2, y1)))
		x, y = unsafe.Add(x, 4*sx), unsafe.Add(y, 4*sy)
	}
	if n == 4 {
		x0, y0 := complexPairAtSIMD(x, 0, sx), complexPairAtSIMD(y, 0, sy)
		x1, y1 := complexPairAtSIMD(x, 2*sx, sx), complexPairAtSIMD(y, 2*sy, sy)
		r0, i0 = r0.Add(x0.Mul(y0)), i0.Add(x0.Mul(y0.ConcatPermuteScalarsGrouped(1, 2, y0)))
		r1, i1 = r1.Add(x1.Mul(y1)), i1.Add(x1.Mul(y1.ConcatPermuteScalarsGrouped(1, 2, y1)))
		n = 0
	}
	r0, i0 = r0.Add(r1), i0.Add(i1)
	if n >= 2 {
		xv, yv := complexPairAtSIMD(x, 0, sx), complexPairAtSIMD(y, 0, sy)
		r0, i0 = r0.Add(xv.Mul(yv)), i0.Add(xv.Mul(yv.ConcatPermuteScalarsGrouped(1, 2, yv)))
		n -= 2
		if n != 0 {
			x, y = unsafe.Add(x, 2*sx), unsafe.Add(y, 2*sy)
		}
	}
	r, im := r0.GetLo().Add(r0.GetHi()), i0.GetLo().Add(i0.GetHi())
	if n != 0 {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(x))
		yv := archsimd.LoadFloat64x2Array((*[2]float64)(y))
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
