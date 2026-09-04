// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"simd/archsimd"
	"unsafe"
)

// AxpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func AxpyUnitary(alpha float64, x, y []float64) {
	if len(y) < len(x) || !axpyUnitarySIMDCompatible(x, y) {
		axpyUnitaryScalar(alpha, x, y)
		return
	}

	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	a := archsimd.BroadcastFloat64x2(alpha)
	var i int
	for ; i+8 <= len(x); i += 8 {
		x0 := loadAxpyFloat64x2(xData, i)
		x1 := loadAxpyFloat64x2(xData, i+2)
		x2 := loadAxpyFloat64x2(xData, i+4)
		x3 := loadAxpyFloat64x2(xData, i+6)
		y0 := loadAxpyFloat64x2(yData, i)
		y1 := loadAxpyFloat64x2(yData, i+2)
		y2 := loadAxpyFloat64x2(yData, i+4)
		y3 := loadAxpyFloat64x2(yData, i+6)

		storeAxpyFloat64x2(yData, i, a.MulAdd(x0, y0))
		storeAxpyFloat64x2(yData, i+2, a.MulAdd(x1, y1))
		storeAxpyFloat64x2(yData, i+4, a.MulAdd(x2, y2))
		storeAxpyFloat64x2(yData, i+6, a.MulAdd(x3, y3))
	}
	for ; i+2 <= len(x); i += 2 {
		xv := loadAxpyFloat64x2(xData, i)
		yv := loadAxpyFloat64x2(yData, i)
		storeAxpyFloat64x2(yData, i, a.MulAdd(xv, yv))
	}
	for ; i < len(x); i++ {
		y[i] += alpha * x[i]
	}
}

func loadAxpyFloat64x2(data unsafe.Pointer, i int) archsimd.Float64x2 {
	return archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(data, uintptr(i)*unsafe.Sizeof(float64(0)))))
}

func storeAxpyFloat64x2(data unsafe.Pointer, i int, v archsimd.Float64x2) {
	v.StoreArray((*[2]float64)(unsafe.Add(data, uintptr(i)*unsafe.Sizeof(float64(0)))))
}

func axpyUnitarySIMDCompatible(x, y []float64) bool {
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

func axpyUnitaryScalar(alpha float64, x, y []float64) {
	for i, v := range x {
		y[i] += alpha * v
	}
}

// AxpyUnitaryTo is
//
//	for i, v := range x {
//		dst[i] = alpha*v + y[i]
//	}
func AxpyUnitaryTo(dst []float64, alpha float64, x, y []float64) {
	if len(x) < 8 || len(dst) < len(x) || len(y) < len(x) ||
		!axpyUnitarySIMDCompatible(x, dst) || !axpyUnitarySIMDCompatible(y, dst) {
		axpyUnitaryToScalar(dst, alpha, x, y)
		return
	}

	dstData := unsafe.Pointer(unsafe.SliceData(dst))
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	a := archsimd.BroadcastFloat64x2(alpha)
	var i int
	for ; i+8 <= len(x); i += 8 {
		x0 := loadAxpyFloat64x2(xData, i)
		x1 := loadAxpyFloat64x2(xData, i+2)
		x2 := loadAxpyFloat64x2(xData, i+4)
		x3 := loadAxpyFloat64x2(xData, i+6)
		y0 := loadAxpyFloat64x2(yData, i)
		y1 := loadAxpyFloat64x2(yData, i+2)
		y2 := loadAxpyFloat64x2(yData, i+4)
		y3 := loadAxpyFloat64x2(yData, i+6)

		storeAxpyFloat64x2(dstData, i, a.MulAdd(x0, y0))
		storeAxpyFloat64x2(dstData, i+2, a.MulAdd(x1, y1))
		storeAxpyFloat64x2(dstData, i+4, a.MulAdd(x2, y2))
		storeAxpyFloat64x2(dstData, i+6, a.MulAdd(x3, y3))
	}
	for ; i+2 <= len(x); i += 2 {
		xv := loadAxpyFloat64x2(xData, i)
		yv := loadAxpyFloat64x2(yData, i)
		storeAxpyFloat64x2(dstData, i, a.MulAdd(xv, yv))
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func axpyUnitaryToScalar(dst []float64, alpha float64, x, y []float64) {
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
func AxpyInc(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
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
func AxpyIncTo(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	for i := 0; i < int(n); i++ {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}
