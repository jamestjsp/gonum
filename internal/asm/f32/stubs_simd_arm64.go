// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f32

import (
	"simd/archsimd"
	"unsafe"
)

// AxpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func AxpyUnitary(alpha float32, x, y []float32) {
	if len(y) < len(x) || !simdSlicesIndependent(x, y) {
		axpyUnitaryScalar(alpha, x, y)
		return
	}
	a := archsimd.BroadcastFloat32x4(alpha)
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+16 <= len(x); i += 16 {
		storeFloat32x4(yData, i, a.MulAdd(loadFloat32x4(xData, i), loadFloat32x4(yData, i)))
		storeFloat32x4(yData, i+4, a.MulAdd(loadFloat32x4(xData, i+4), loadFloat32x4(yData, i+4)))
		storeFloat32x4(yData, i+8, a.MulAdd(loadFloat32x4(xData, i+8), loadFloat32x4(yData, i+8)))
		storeFloat32x4(yData, i+12, a.MulAdd(loadFloat32x4(xData, i+12), loadFloat32x4(yData, i+12)))
	}
	for ; i+4 <= len(x); i += 4 {
		storeFloat32x4(yData, i, a.MulAdd(loadFloat32x4(xData, i), loadFloat32x4(yData, i)))
	}
	for ; i < len(x); i++ {
		y[i] += alpha * x[i]
	}
}

// AxpyUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha*v + y[i]
//	}
func AxpyUnitaryTo(dst []float32, alpha float32, x, y []float32) {
	if len(x) < 16 || len(dst) < len(x) || len(y) < len(x) || !simdDestinationCompatible(dst, x, len(x)) || !simdDestinationCompatible(dst, y, len(x)) {
		axpyUnitaryToScalar(dst, alpha, x, y)
		return
	}
	a := archsimd.BroadcastFloat32x4(alpha)
	dstData := unsafe.Pointer(unsafe.SliceData(dst))
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	i := 0
	for ; i+16 <= len(x); i += 16 {
		storeFloat32x4(dstData, i, a.MulAdd(loadFloat32x4(xData, i), loadFloat32x4(yData, i)))
		storeFloat32x4(dstData, i+4, a.MulAdd(loadFloat32x4(xData, i+4), loadFloat32x4(yData, i+4)))
		storeFloat32x4(dstData, i+8, a.MulAdd(loadFloat32x4(xData, i+8), loadFloat32x4(yData, i+8)))
		storeFloat32x4(dstData, i+12, a.MulAdd(loadFloat32x4(xData, i+12), loadFloat32x4(yData, i+12)))
	}
	for ; i+4 <= len(x); i += 4 {
		storeFloat32x4(dstData, i, a.MulAdd(loadFloat32x4(xData, i), loadFloat32x4(yData, i)))
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyInc(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func AxpyIncTo(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

func DotUnitary(x, y []float32) (sum float32) {
	if len(x) < 16 {
		for i, v := range x {
			sum += y[i] * v
		}
		return sum
	}
	var sum0, sum1, sum2, sum3 archsimd.Float32x4
	i := 0
	for ; i+16 <= len(x); i += 16 {
		sum0 = archsimd.LoadFloat32x4(x[i:]).MulAdd(archsimd.LoadFloat32x4(y[i:]), sum0)
		sum1 = archsimd.LoadFloat32x4(x[i+4:]).MulAdd(archsimd.LoadFloat32x4(y[i+4:]), sum1)
		sum2 = archsimd.LoadFloat32x4(x[i+8:]).MulAdd(archsimd.LoadFloat32x4(y[i+8:]), sum2)
		sum3 = archsimd.LoadFloat32x4(x[i+12:]).MulAdd(archsimd.LoadFloat32x4(y[i+12:]), sum3)
	}
	sum0 = sum0.Add(sum1).Add(sum2.Add(sum3))
	for ; i+4 <= len(x); i += 4 {
		sum0 = archsimd.LoadFloat32x4(x[i:]).MulAdd(archsimd.LoadFloat32x4(y[i:]), sum0)
	}
	pairs := sum0.ConcatAddPairs(sum0)
	sum = pairs.ConcatAddPairs(pairs).GetElem(0)
	for ; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}

func DotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float32) {
	for i := 0; i < int(n); i++ {
		sum += y[iy] * x[ix]
		ix += incX
		iy += incY
	}
	return sum
}

func DdotUnitary(x, y []float32) (sum float64) {
	if len(x) < 16 {
		for i, v := range x {
			sum += float64(y[i]) * float64(v)
		}
		return sum
	}
	var sum0, sum1, sum2, sum3 archsimd.Float64x2
	i := 0
	for ; i+8 <= len(x); i += 8 {
		x0 := archsimd.LoadFloat32x4(x[i:])
		y0 := archsimd.LoadFloat32x4(y[i:])
		x1 := archsimd.LoadFloat32x4(x[i+4:])
		y1 := archsimd.LoadFloat32x4(y[i+4:])
		sum0 = x0.ConvertLo2ToFloat64().MulAdd(y0.ConvertLo2ToFloat64(), sum0)
		sum1 = x0.HiToLo().ConvertLo2ToFloat64().MulAdd(y0.HiToLo().ConvertLo2ToFloat64(), sum1)
		sum2 = x1.ConvertLo2ToFloat64().MulAdd(y1.ConvertLo2ToFloat64(), sum2)
		sum3 = x1.HiToLo().ConvertLo2ToFloat64().MulAdd(y1.HiToLo().ConvertLo2ToFloat64(), sum3)
	}
	sum0 = sum0.Add(sum1).Add(sum2.Add(sum3))
	for ; i+4 <= len(x); i += 4 {
		xv := archsimd.LoadFloat32x4(x[i:])
		yv := archsimd.LoadFloat32x4(y[i:])
		sum0 = xv.ConvertLo2ToFloat64().MulAdd(yv.ConvertLo2ToFloat64(), sum0)
		sum0 = xv.HiToLo().ConvertLo2ToFloat64().MulAdd(yv.HiToLo().ConvertLo2ToFloat64(), sum0)
	}
	sum = sum0.ConcatAddPairs(sum0).GetElem(0)
	for ; i < len(x); i++ {
		sum += float64(x[i]) * float64(y[i])
	}
	return sum
}

func DdotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float64) {
	for i := 0; i < int(n); i++ {
		sum += float64(y[iy]) * float64(x[ix])
		ix += incX
		iy += incY
	}
	return sum
}

func Sum(x []float32) (sum float32) {
	for _, v := range x {
		sum += v
	}
	return sum
}

func axpyUnitaryScalar(alpha float32, x, y []float32) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

func axpyUnitaryToScalar(dst []float32, alpha float32, x, y []float32) {
	for i, v := range x {
		dst[i] = alpha*v + y[i]
	}
}

func loadFloat32x4(data unsafe.Pointer, i int) archsimd.Float32x4 {
	return archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(data, uintptr(i)*unsafe.Sizeof(float32(0)))))
}

func storeFloat32x4(data unsafe.Pointer, i int, v archsimd.Float32x4) {
	v.StoreArray((*[4]float32)(unsafe.Add(data, uintptr(i)*unsafe.Sizeof(float32(0)))))
}

func simdSlicesIndependent(x, y []float32) bool {
	if len(x) == 0 {
		return true
	}
	xStart := uintptr(unsafe.Pointer(unsafe.SliceData(x)))
	yStart := uintptr(unsafe.Pointer(unsafe.SliceData(y)))
	if xStart == yStart {
		return true
	}
	n := uintptr(len(x)) * unsafe.Sizeof(x[0])
	return xStart+n <= yStart || yStart+n <= xStart
}

func simdDestinationCompatible(dst, src []float32, n int) bool {
	if n == 0 {
		return true
	}
	dstStart := uintptr(unsafe.Pointer(unsafe.SliceData(dst)))
	srcStart := uintptr(unsafe.Pointer(unsafe.SliceData(src)))
	if dstStart == srcStart {
		return true
	}
	bytes := uintptr(n) * unsafe.Sizeof(src[0])
	return dstStart+bytes <= srcStart || srcStart+bytes <= dstStart
}
