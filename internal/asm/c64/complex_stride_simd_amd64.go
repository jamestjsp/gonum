// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

// General strides use the smallest native vector: a complex128 value or a pair
// of complex64 values. This avoids staging separate component arrays, and works
// for forward, reverse, and repeated read indices without requiring gathers.
func complexNativeSIMD() bool {
	// The native broadcasts require AVX2 even for 128-bit vectors.
	if !archsimd.X86.AVX2() {
		return false
	}
	switch simd.BroadcastFloat64s(0).ToArch().(type) {
	case archsimd.Float64x2, archsimd.Float64x4, archsimd.Float64x8:
		return true
	}
	return false
}

func complexLoadPairSIMD(x []complex64, ix, inc uintptr) archsimd.Float32x4 {
	return archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Pointer(&x[ix]))).SetElem(1, *(*uint64)(unsafe.Pointer(&x[ix+inc]))).AsFloat32x4()
}

func complexAxpyIncNativeSIMD(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	ar, ai := archsimd.BroadcastFloat32x4(real(alpha)), archsimd.BroadcastFloat32x4(imag(alpha))
	for ; n >= 2; n -= 2 {
		xv, yv := complexLoadPairSIMD(x, ix, incX), complexLoadPairSIMD(y, iy, incY)
		result := xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalars(1, 0, 7, 6, xv).Mul(ai)).Add(yv).AsUint64x2()
		*(*uint64)(unsafe.Pointer(&dst[idst])) = result.GetElem(0)
		*(*uint64)(unsafe.Pointer(&dst[idst+incDst])) = result.GetElem(1)
		ix += 2 * incX
		iy += 2 * incY
		idst += 2 * incDst
	}
	if n > 0 {
		dst[idst] = alpha*x[ix] + y[iy]
	}
}

func complexDotIncNativeSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr, conjugate bool) complex64 {
	originalN, originalX, originalY := n, ix, iy
	sign := archsimd.BroadcastUint64x2(1 << 63).AsUint32x4()
	conjugateSign := archsimd.BroadcastUint64x2(0).AsUint32x4()
	if conjugate {
		conjugateSign = sign
	}
	sum0, sum1 := archsimd.BroadcastFloat32x4(0), archsimd.BroadcastFloat32x4(0)
	for ; n >= 4; n -= 4 {
		x0, y0 := complexLoadPairSIMD(x, ix, incX).ToBits().Xor(conjugateSign).BitsToFloat32(), complexLoadPairSIMD(y, iy, incY)
		ix += 2 * incX
		iy += 2 * incY
		x1, y1 := complexLoadPairSIMD(x, ix, incX).ToBits().Xor(conjugateSign).BitsToFloat32(), complexLoadPairSIMD(y, iy, incY)
		ix += 2 * incX
		iy += 2 * incY
		p0 := x0.Mul(y0).ToBits().Xor(sign).BitsToFloat32()
		q0 := x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0))
		p1 := x1.Mul(y1).ToBits().Xor(sign).BitsToFloat32()
		q1 := x1.Mul(y1.ConcatPermuteScalars(1, 0, 7, 6, y1))
		sum0 = sum0.Add(p0.ConcatAddPairs(q0))
		sum1 = sum1.Add(p1.ConcatAddPairs(q1))
	}
	sum0 = sum0.Add(sum1)
	if n >= 2 {
		xv, yv := complexLoadPairSIMD(x, ix, incX).ToBits().Xor(conjugateSign).BitsToFloat32(), complexLoadPairSIMD(y, iy, incY)
		p := xv.Mul(yv).ToBits().Xor(sign).BitsToFloat32()
		q := xv.Mul(yv.ConcatPermuteScalars(1, 0, 7, 6, yv))
		sum0 = sum0.Add(p.ConcatAddPairs(q))
		ix += 2 * incX
		iy += 2 * incY
		n -= 2
	}
	sum := complex(sum0.GetElem(0)+sum0.GetElem(1), sum0.GetElem(2)+sum0.GetElem(3))
	if n > 0 {
		v := x[ix]
		if conjugate {
			v = conj64(v)
		}
		sum += v * y[iy]
	}
	if !math.IsNaN(float64(real(sum))) && !math.IsNaN(float64(imag(sum))) && !math.IsInf(float64(real(sum)), 0) && !math.IsInf(float64(imag(sum)), 0) {
		return sum
	}
	// Reassociation can overflow both the native and sequential sums. Retry
	// the established portable grouping before per-element recovery.
	sum = portableDotIncSIMD(x, y, originalN, incX, incY, originalX, originalY, conjugate)
	if !math.IsNaN(float64(real(sum))) && !math.IsNaN(float64(imag(sum))) && !math.IsInf(float64(real(sum)), 0) && !math.IsInf(float64(imag(sum)), 0) {
		return sum
	}
	sum = 0
	ix, iy = originalX, originalY
	for ; originalN > 0; originalN-- {
		v := x[ix]
		if conjugate {
			v = conj64(v)
		}
		sum += v * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

// A short contiguous dot needs neither packing nor wide-vector scalar tails.
func complexDotShortNativeSIMD(x, y []complex64, conjugate bool) complex64 {
	y = y[:len(x):len(x)]
	sign := archsimd.BroadcastUint64x2(1 << 63).AsUint32x4()
	conjugateSign := archsimd.BroadcastUint64x2(0).AsUint32x4()
	if conjugate {
		conjugateSign = sign
	}
	sum0, sum1 := archsimd.BroadcastFloat32x4(0), archsimd.BroadcastFloat32x4(0)
	for len(x) >= 4 {
		x0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[0]))).ToBits().Xor(conjugateSign).BitsToFloat32()
		y0 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[0])))
		x1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[2]))).ToBits().Xor(conjugateSign).BitsToFloat32()
		y1 := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[2])))
		p0 := x0.Mul(y0).ToBits().Xor(sign).BitsToFloat32()
		q0 := x0.Mul(y0.ConcatPermuteScalars(1, 0, 7, 6, y0))
		p1 := x1.Mul(y1).ToBits().Xor(sign).BitsToFloat32()
		q1 := x1.Mul(y1.ConcatPermuteScalars(1, 0, 7, 6, y1))
		sum0 = sum0.Add(p0.ConcatAddPairs(q0))
		sum1 = sum1.Add(p1.ConcatAddPairs(q1))
		x, y = x[4:], y[4:]
	}
	sum0 = sum0.Add(sum1)
	if len(x) >= 2 {
		xv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[0]))).ToBits().Xor(conjugateSign).BitsToFloat32()
		yv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[0])))
		p := xv.Mul(yv).ToBits().Xor(sign).BitsToFloat32()
		q := xv.Mul(yv.ConcatPermuteScalars(1, 0, 7, 6, yv))
		sum0 = sum0.Add(p.ConcatAddPairs(q))
		x, y = x[2:], y[2:]
	}
	sum0 = sum0.ConcatAddPairs(sum0)
	sum := complex(sum0.GetElem(0), sum0.GetElem(1))
	if len(x) > 0 {
		v := x[0]
		if conjugate {
			v = conj64(v)
		}
		sum += v * y[0]
	}
	return sum
}

func complexAxpyTailNativeSIMD(dst []complex64, alpha complex64, x, y []complex64) int {
	ar, ai := archsimd.BroadcastFloat32x4(real(alpha)), archsimd.BroadcastFloat32x4(imag(alpha))
	n := len(x)
	dst, y = dst[:n:n], y[:n:n]
	for len(x) >= 2 {
		xv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&x[0])))
		yv := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Pointer(&y[0])))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalars(1, 0, 7, 6, xv).Mul(ai)).Add(yv).StoreArray((*[4]float32)(unsafe.Pointer(&dst[0])))
		x, y, dst = x[2:], y[2:], dst[2:]
	}
	return n - len(x)
}
