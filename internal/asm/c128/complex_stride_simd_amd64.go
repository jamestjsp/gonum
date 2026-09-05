// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

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

func complexAxpyIncNativeSIMD(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	ar := archsimd.BroadcastFloat64x2(real(alpha))
	ai := archsimd.BroadcastFloat64x2(imag(alpha))
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix])))
		yv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&y[iy])))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai)).Add(yv).StoreArray((*[2]float64)(unsafe.Pointer(&dst[idst])))
		ix += incX
		iy += incY
		idst += incDst
	}
}

func complexDotIncNativeSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr, conjugate bool) complex128 {
	originalN, originalX, originalY := n, ix, iy
	sign := archsimd.BroadcastUint64x2(0).SetElem(1, 1<<63)
	conjugateSign := archsimd.BroadcastUint64x2(0)
	if conjugate {
		conjugateSign = sign
	}
	sum0, sum1 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	sum2, sum3 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	for ; n >= 4; n -= 4 {
		x0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix]))).ToBits().Xor(conjugateSign).BitsToFloat64()
		y0 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&y[iy])))
		ix += incX
		iy += incY
		x1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix]))).ToBits().Xor(conjugateSign).BitsToFloat64()
		y1 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&y[iy])))
		ix += incX
		iy += incY
		x2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix]))).ToBits().Xor(conjugateSign).BitsToFloat64()
		y2 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&y[iy])))
		ix += incX
		iy += incY
		x3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix]))).ToBits().Xor(conjugateSign).BitsToFloat64()
		y3 := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&y[iy])))
		ix += incX
		iy += incY
		p0 := x0.Mul(y0).ToBits().Xor(sign).BitsToFloat64()
		q0 := x0.Mul(y0.ConcatPermuteScalars(1, 2, y0))
		p1 := x1.Mul(y1).ToBits().Xor(sign).BitsToFloat64()
		q1 := x1.Mul(y1.ConcatPermuteScalars(1, 2, y1))
		p2 := x2.Mul(y2).ToBits().Xor(sign).BitsToFloat64()
		q2 := x2.Mul(y2.ConcatPermuteScalars(1, 2, y2))
		p3 := x3.Mul(y3).ToBits().Xor(sign).BitsToFloat64()
		q3 := x3.Mul(y3.ConcatPermuteScalars(1, 2, y3))
		sum0 = sum0.Add(p0.ConcatAddPairs(q0))
		sum1 = sum1.Add(p1.ConcatAddPairs(q1))
		sum2 = sum2.Add(p2.ConcatAddPairs(q2))
		sum3 = sum3.Add(p3.ConcatAddPairs(q3))
	}
	sum0 = sum0.Add(sum1).Add(sum2).Add(sum3)
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix]))).ToBits().Xor(conjugateSign).BitsToFloat64()
		yv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&y[iy])))
		p := xv.Mul(yv).ToBits().Xor(sign).BitsToFloat64()
		q := xv.Mul(yv.ConcatPermuteScalars(1, 2, yv))
		sum0 = sum0.Add(p.ConcatAddPairs(q))
		ix += incX
		iy += incY
	}
	sum := complex(sum0.GetElem(0), sum0.GetElem(1))
	if !math.IsNaN(real(sum)) && !math.IsNaN(imag(sum)) && !math.IsInf(real(sum), 0) && !math.IsInf(imag(sum), 0) {
		return sum
	}
	// Reassociation can overflow both the native and sequential sums. Retry
	// the established portable grouping before per-element recovery.
	sum = portableDotIncSIMD(x, y, originalN, incX, incY, originalX, originalY, conjugate)
	if !math.IsNaN(real(sum)) && !math.IsNaN(imag(sum)) && !math.IsInf(real(sum), 0) && !math.IsInf(imag(sum), 0) {
		return sum
	}
	sum = 0
	ix, iy = originalX, originalY
	for ; originalN > 0; originalN-- {
		v := x[ix]
		if conjugate {
			v = conj128(v)
		}
		sum += v * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func complexScalIncNativeSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	ar, ai := archsimd.BroadcastFloat64x2(real(alpha)), archsimd.BroadcastFloat64x2(imag(alpha))
	var ix uintptr
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix])))
		xv.Mul(ar).AddOddSubEven(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai)).StoreArray((*[2]float64)(unsafe.Pointer(&x[ix])))
		ix += inc
	}
}

func complexDscalIncNativeSIMD(alpha float64, x []complex128, n, inc uintptr) {
	a := archsimd.BroadcastFloat64x2(alpha)
	var ix uintptr
	for ; n > 0; n-- {
		xv := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Pointer(&x[ix])))
		xv.Mul(a).StoreArray((*[2]float64)(unsafe.Pointer(&x[ix])))
		ix += inc
	}
}
