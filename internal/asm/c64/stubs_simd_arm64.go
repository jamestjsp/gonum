// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package c64

import (
	"simd/archsimd"
	"unsafe"
)

// AxpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func AxpyUnitary(alpha complex64, x, y []complex64) {
	if len(y) < len(x) || !simdSlicesCompatible(x, y) || len(x) < 8 {
		axpyUnitaryScalar(alpha, x, y)
		return
	}
	ar, ai := complexAlphaVectors(alpha)
	for i := 0; i+2 <= len(x); i += 2 {
		xv := loadComplex64x2(x, i)
		yv := loadComplex64x2(y, i)
		complexScaleVector(xv, ar, ai).Add(yv).StoreArray(complex64Array(y, i))
	}
	if len(x)&1 != 0 {
		i := len(x) - 1
		y[i] += alpha * x[i]
	}
}

// AxpyUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha*v + y[i]
//	}
func AxpyUnitaryTo(dst []complex64, alpha complex64, x, y []complex64) {
	if len(dst) < len(x) || len(y) < len(x) || !simdSlicesCompatible(dst[:len(x)], x) || !simdSlicesCompatible(dst[:len(x)], y) || len(x) < 8 {
		for i, v := range x {
			dst[i] = alpha*v + y[i]
		}
		return
	}
	ar, ai := complexAlphaVectors(alpha)
	for i := 0; i+2 <= len(x); i += 2 {
		xv := loadComplex64x2(x, i)
		yv := loadComplex64x2(y, i)
		complexScaleVector(xv, ar, ai).Add(yv).StoreArray(complex64Array(dst, i))
	}
	if len(x)&1 != 0 {
		i := len(x) - 1
		dst[i] = alpha*x[i] + y[i]
	}
}

// AxpyInc is
//
//	for i := 0; i < int(n); i++ {
//		y[iy] += alpha * x[ix]
//		ix += incX
//		iy += incY
//	}
func AxpyInc(alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
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
func AxpyIncTo(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

// DotcUnitary is
//
//	for i, v := range x {
//		sum += y[i] * conj(v)
//	}
//	return sum
func DotcUnitary(x, y []complex64) (sum complex64) {
	return dotUnitarySIMD(x, y, true)
}

// DotcInc is
//
//	for i := 0; i < int(n); i++ {
//		sum += y[iy] * conj(x[ix])
//		ix += incX
//		iy += incY
//	}
//	return sum
func DotcInc(x, y []complex64, n, incX, incY, ix, iy uintptr) (sum complex64) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * conj(x[ix])
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
func DotuUnitary(x, y []complex64) (sum complex64) {
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
func DotuInc(x, y []complex64, n, incX, incY, ix, iy uintptr) (sum complex64) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * x[ix]
		ix += incX
		iy += incY
	}
	return sum
}

func dotUnitarySIMD(x, y []complex64, conjugate bool) (sum complex64) {
	if len(x) < 16 {
		for i, xv := range x {
			if conjugate {
				xv = conj(xv)
			}
			sum += xv * y[i]
		}
		return sum
	}
	var sum0, sum1, sum2, sum3 archsimd.Float32x4
	var i int
	for ; i+8 <= len(x); i += 8 {
		sum0 = sum0.Add(complexProductVector(loadComplex64x2(x, i), loadComplex64x2(y, i), conjugate))
		sum1 = sum1.Add(complexProductVector(loadComplex64x2(x, i+2), loadComplex64x2(y, i+2), conjugate))
		sum2 = sum2.Add(complexProductVector(loadComplex64x2(x, i+4), loadComplex64x2(y, i+4), conjugate))
		sum3 = sum3.Add(complexProductVector(loadComplex64x2(x, i+6), loadComplex64x2(y, i+6), conjugate))
	}
	acc := sum0.Add(sum1).Add(sum2.Add(sum3))
	for ; i+2 <= len(x); i += 2 {
		acc = acc.Add(complexProductVector(loadComplex64x2(x, i), loadComplex64x2(y, i), conjugate))
	}
	acc = acc.Add(acc.HiToLo())
	sum = complex(acc.GetElem(0), acc.GetElem(1))
	if i < len(x) {
		xv := x[i]
		if conjugate {
			xv = conj(xv)
		}
		sum += xv * y[i]
	}
	return sum
}

func complexProductVector(x, y archsimd.Float32x4, conjugate bool) archsimd.Float32x4 {
	xb := x.ToBits()
	xr := xb.InterleaveEven(xb).BitsToFloat32()
	xi := xb.InterleaveOdd(xb).BitsToFloat32()
	yb := y.ToBits()
	yswap := yb.InterleaveOdd(yb).InterleaveEven(yb.InterleaveEven(yb)).BitsToFloat32()
	if conjugate {
		yswap = yswap.ToBits().Xor(archsimd.LoadUint32x4([]uint32{0, 1 << 31, 0, 1 << 31})).BitsToFloat32()
	} else {
		yswap = yswap.ToBits().Xor(archsimd.LoadUint32x4([]uint32{1 << 31, 0, 1 << 31, 0})).BitsToFloat32()
	}
	return xr.Mul(y).Add(xi.Mul(yswap))
}

func complexAlphaVectors(alpha complex64) (realPart, imagPart archsimd.Float32x4) {
	ar, ai := real(alpha), imag(alpha)
	realPart = archsimd.LoadFloat32x4([]float32{ar, ai, ar, ai})
	imagPart = archsimd.LoadFloat32x4([]float32{-ai, ar, -ai, ar})
	return realPart, imagPart
}

func complexScaleVector(x, realPart, imagPart archsimd.Float32x4) archsimd.Float32x4 {
	xb := x.ToBits()
	return xb.InterleaveEven(xb).BitsToFloat32().Mul(realPart).Add(xb.InterleaveOdd(xb).BitsToFloat32().Mul(imagPart))
}

func loadComplex64x2(x []complex64, i int) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array(complex64Array(x, i))
}

func complex64Array(x []complex64, i int) *[4]float32 {
	return (*[4]float32)(unsafe.Add(unsafe.Pointer(unsafe.SliceData(x)), uintptr(i)*unsafe.Sizeof(complex64(0))))
}

func simdSlicesCompatible(a, b []complex64) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if aStart == bStart {
		return true
	}
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(complex64(0))
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(complex64(0))
	return aEnd <= bStart || bEnd <= aStart
}

func axpyUnitaryScalar(alpha complex64, x, y []complex64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}
