// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package c128

import (
	"math/cmplx"
	"simd/archsimd"
	"unsafe"
)

// AxpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func AxpyUnitary(alpha complex128, x, y []complex128) {
	if len(y) < len(x) || !simdSlicesCompatible(x, y) || len(x) < 4 {
		axpyUnitaryScalar(alpha, x, y)
		return
	}
	ar, ai := complexAlphaVectors(alpha)
	for i := range x {
		xv := loadComplex128(x, i)
		yv := loadComplex128(y, i)
		complexScaleVector(xv, ar, ai).Add(yv).StoreArray(complex128Array(y, i))
	}
}

// AxpyUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha*v + y[i]
//	}
func AxpyUnitaryTo(dst []complex128, alpha complex128, x, y []complex128) {
	for i, v := range x {
		dst[i] = alpha*v + y[i]
	}
}

// AxpyInc is
//
//	for i := 0; i < int(n); i++ {
//		y[iy] += alpha * x[ix]
//		ix += incX
//		iy += incY
//	}
func AxpyInc(alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

// AxpyIncTo is
//
//	for i := 0; i < int(n); i++ {
//		dst[idst] = alpha*x[ix] + y[iy]
//		ix += incX
//		iy += incY
//		idst += incDst
//	}
func AxpyIncTo(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

// DscalUnitary is
//
//	for i, v := range x {
//		x[i] = complex(real(v)*alpha, imag(v)*alpha)
//	}
func DscalUnitary(alpha float64, x []complex128) {
	a := archsimd.BroadcastFloat64x2(alpha)
	for i := range x {
		loadComplex128(x, i).Mul(a).StoreArray(complex128Array(x, i))
	}
}

// DscalInc is
//
//	for i := 0; i < int(n); i++ {
//		x[ix] = complex(real(x[ix])*alpha, imag(x[ix])*alpha)
//		ix += inc
//	}
func DscalInc(alpha float64, x []complex128, n, inc uintptr) {
	var ix uintptr
	for i := 0; i < int(n); i++ {
		x[ix] = complex(real(x[ix])*alpha, imag(x[ix])*alpha)
		ix += inc
	}
}

// ScalInc is
//
//	for i := 0; i < int(n); i++ {
//		x[ix] *= alpha
//		ix += incX
//	}
func ScalInc(alpha complex128, x []complex128, n, inc uintptr) {
	var ix uintptr
	for i := 0; i < int(n); i++ {
		x[ix] *= alpha
		ix += inc
	}
}

// ScalUnitary is
//
//	for i := range x {
//		x[i] *= alpha
//	}
func ScalUnitary(alpha complex128, x []complex128) {
	for i := range x {
		x[i] *= alpha
	}
}

// DotcUnitary is
//
//	for i, v := range x {
//		sum += y[i] * cmplx.Conj(v)
//	}
//	return sum
func DotcUnitary(x, y []complex128) (sum complex128) {
	if len(x) < 128 {
		for i, v := range x {
			sum += y[i] * cmplx.Conj(v)
		}
		return sum
	}
	return dotUnitarySIMD(x, y, true)
}

// DotcInc is
//
//	for i := 0; i < int(n); i++ {
//		sum += y[iy] * cmplx.Conj(x[ix])
//		ix += incX
//		iy += incY
//	}
//	return sum
func DotcInc(x, y []complex128, n, incX, incY, ix, iy uintptr) (sum complex128) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * cmplx.Conj(x[ix])
		ix += incX
		iy += incY
	}
	return sum
}

// DotuUnitary is
//
//	for i, v := range x {
//		sum += y[i] * v
//	}
//	return sum
func DotuUnitary(x, y []complex128) (sum complex128) {
	if len(x) < 128 {
		for i, v := range x {
			sum += y[i] * v
		}
		return sum
	}
	return dotUnitarySIMD(x, y, false)
}

// DotuInc is
//
//	for i := 0; i < int(n); i++ {
//		sum += y[iy] * x[ix]
//		ix += incX
//		iy += incY
//	}
//	return sum
func DotuInc(x, y []complex128, n, incX, incY, ix, iy uintptr) (sum complex128) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * x[ix]
		ix += incX
		iy += incY
	}
	return sum
}

func dotUnitarySIMD(x, y []complex128, conjugate bool) (sum complex128) {
	var sum0, sum1, sum2, sum3 archsimd.Float64x2
	var i int
	for ; i+4 <= len(x); i += 4 {
		sum0 = sum0.Add(complexProductVector(loadComplex128(x, i), loadComplex128(y, i), conjugate))
		sum1 = sum1.Add(complexProductVector(loadComplex128(x, i+1), loadComplex128(y, i+1), conjugate))
		sum2 = sum2.Add(complexProductVector(loadComplex128(x, i+2), loadComplex128(y, i+2), conjugate))
		sum3 = sum3.Add(complexProductVector(loadComplex128(x, i+3), loadComplex128(y, i+3), conjugate))
	}
	acc := sum0.Add(sum1).Add(sum2.Add(sum3))
	for ; i < len(x); i++ {
		acc = acc.Add(complexProductVector(loadComplex128(x, i), loadComplex128(y, i), conjugate))
	}
	return complex(acc.GetElem(0), acc.GetElem(1))
}

func complexProductVector(x, y archsimd.Float64x2, conjugate bool) archsimd.Float64x2 {
	xb := x.ToBits()
	xr := xb.InterleaveEven(xb).BitsToFloat64()
	xi := xb.InterleaveOdd(xb).BitsToFloat64()
	yb := y.ToBits().ReshapeToUint8s()
	yswap := yb.ConcatShiftBytesRight(yb, 8).ReshapeToUint64s().BitsToFloat64()
	if conjugate {
		yswap = yswap.ToBits().Xor(archsimd.LoadUint64x2([]uint64{0, 1 << 63})).BitsToFloat64()
	} else {
		yswap = yswap.ToBits().Xor(archsimd.LoadUint64x2([]uint64{1 << 63, 0})).BitsToFloat64()
	}
	return xr.Mul(y).Add(xi.Mul(yswap))
}

func complexAlphaVectors(alpha complex128) (realPart, imagPart archsimd.Float64x2) {
	ar, ai := real(alpha), imag(alpha)
	realPart = archsimd.LoadFloat64x2([]float64{ar, ai})
	imagPart = archsimd.LoadFloat64x2([]float64{-ai, ar})
	return realPart, imagPart
}

func complexScaleVector(x, realPart, imagPart archsimd.Float64x2) archsimd.Float64x2 {
	xb := x.ToBits()
	return xb.InterleaveEven(xb).BitsToFloat64().Mul(realPart).Add(xb.InterleaveOdd(xb).BitsToFloat64().Mul(imagPart))
}

func loadComplex128(x []complex128, i int) archsimd.Float64x2 {
	return archsimd.LoadFloat64x2Array(complex128Array(x, i))
}

func complex128Array(x []complex128, i int) *[2]float64 {
	return (*[2]float64)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(x)), uintptr(i)*unsafe.Sizeof(complex128(0))))
}

func simdSlicesCompatible(a, b []complex128) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if aStart == bStart {
		return true
	}
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(complex128(0))
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(complex128(0))
	return aEnd <= bStart || bEnd <= aStart
}

func axpyUnitaryScalar(alpha complex128, x, y []complex128) {
	for i, v := range x {
		y[i] += alpha * v
	}
}
