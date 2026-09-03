// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

// DasumUnitary returns the sum of the absolute values of x.
func DasumUnitary(x []float64) float64 {
	if len(x) < 16 {
		return dasumUnitaryScalar(x)
	}

	data := unsafe.Pointer(unsafe.SliceData(x))
	var sum0, sum1, sum2, sum3 archsimd.Float64x2
	var i int
	for ; i+8 <= len(x); i += 8 {
		sum0 = sum0.Add(loadLevel1Float64x2(data, i).Abs())
		sum1 = sum1.Add(loadLevel1Float64x2(data, i+2).Abs())
		sum2 = sum2.Add(loadLevel1Float64x2(data, i+4).Abs())
		sum3 = sum3.Add(loadLevel1Float64x2(data, i+6).Abs())
	}
	sum0 = sum0.Add(sum1)
	sum2 = sum2.Add(sum3)
	sum0 = sum0.Add(sum2)
	sum := sum0.GetElem(0) + sum0.GetElem(1)
	for ; i < len(x); i++ {
		sum += math.Abs(x[i])
	}
	return sum
}

func dasumUnitaryScalar(x []float64) (sum float64) {
	for _, v := range x {
		sum += math.Abs(v)
	}
	return sum
}

// SwapUnitary exchanges corresponding elements of x and y.
func SwapUnitary(x, y []float64) {
	if len(x) < 16 || len(y) < len(x) || !level1SIMDCompatible(x, y) {
		swapUnitaryScalar(x, y)
		return
	}
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	var i int
	for ; i+8 <= len(x); i += 8 {
		x0 := loadLevel1Float64x2(xData, i)
		x1 := loadLevel1Float64x2(xData, i+2)
		x2 := loadLevel1Float64x2(xData, i+4)
		x3 := loadLevel1Float64x2(xData, i+6)
		y0 := loadLevel1Float64x2(yData, i)
		y1 := loadLevel1Float64x2(yData, i+2)
		y2 := loadLevel1Float64x2(yData, i+4)
		y3 := loadLevel1Float64x2(yData, i+6)
		storeLevel1Float64x2(xData, i, y0)
		storeLevel1Float64x2(xData, i+2, y1)
		storeLevel1Float64x2(xData, i+4, y2)
		storeLevel1Float64x2(xData, i+6, y3)
		storeLevel1Float64x2(yData, i, x0)
		storeLevel1Float64x2(yData, i+2, x1)
		storeLevel1Float64x2(yData, i+4, x2)
		storeLevel1Float64x2(yData, i+6, x3)
	}
	for ; i < len(x); i++ {
		x[i], y[i] = y[i], x[i]
	}
}

func swapUnitaryScalar(x, y []float64) {
	for i, v := range x {
		x[i], y[i] = y[i], v
	}
}

// RotUnitary applies a plane rotation to x and y.
func RotUnitary(x, y []float64, c, s float64) {
	if len(x) < 16 || len(y) < len(x) || !level1SIMDCompatible(x, y) {
		rotUnitaryScalar(x, y, c, s)
		return
	}
	cv := archsimd.BroadcastFloat64x2(c)
	sv := archsimd.BroadcastFloat64x2(s)
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	var i int
	for ; i+4 <= len(x); i += 4 {
		x0 := loadLevel1Float64x2(xData, i)
		x1 := loadLevel1Float64x2(xData, i+2)
		y0 := loadLevel1Float64x2(yData, i)
		y1 := loadLevel1Float64x2(yData, i+2)
		storeLevel1Float64x2(xData, i, x0.Mul(cv).Add(y0.Mul(sv)))
		storeLevel1Float64x2(xData, i+2, x1.Mul(cv).Add(y1.Mul(sv)))
		storeLevel1Float64x2(yData, i, y0.Mul(cv).Sub(x0.Mul(sv)))
		storeLevel1Float64x2(yData, i+2, y1.Mul(cv).Sub(x1.Mul(sv)))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = c*vx+s*vy, c*vy-s*vx
	}
}

func rotUnitaryScalar(x, y []float64, c, s float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = c*vx+s*vy, c*vy-s*vx
	}
}

// RotmUnitaryRescaling applies a full modified plane rotation to x and y.
func RotmUnitaryRescaling(x, y []float64, h11, h12, h21, h22 float64) {
	if len(x) < 16 || len(y) < len(x) || !level1SIMDCompatible(x, y) {
		rotmUnitaryRescalingScalar(x, y, h11, h12, h21, h22)
		return
	}
	transformUnitary(
		x, y,
		archsimd.BroadcastFloat64x2(h11),
		archsimd.BroadcastFloat64x2(h12),
		archsimd.BroadcastFloat64x2(h21),
		archsimd.BroadcastFloat64x2(h22),
	)
}

func rotmUnitaryRescalingScalar(x, y []float64, h11, h12, h21, h22 float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = float64(vx*h11)+float64(vy*h12), float64(vx*h21)+float64(vy*h22)
	}
}

// RotmUnitaryOffDiagonal applies an off-diagonal modified plane rotation.
func RotmUnitaryOffDiagonal(x, y []float64, h12, h21 float64) {
	if len(x) < 16 || len(y) < len(x) || !level1SIMDCompatible(x, y) {
		rotmUnitaryOffDiagonalScalar(x, y, h12, h21)
		return
	}
	transformOffDiagonalUnitary(x, y, archsimd.BroadcastFloat64x2(h12), archsimd.BroadcastFloat64x2(h21))
}

func rotmUnitaryOffDiagonalScalar(x, y []float64, h12, h21 float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx+float64(vy*h12), float64(vx*h21)+vy
	}
}

// RotmUnitaryDiagonal applies a diagonal modified plane rotation.
func RotmUnitaryDiagonal(x, y []float64, h11, h22 float64) {
	if len(x) < 16 || len(y) < len(x) || !level1SIMDCompatible(x, y) {
		rotmUnitaryDiagonalScalar(x, y, h11, h22)
		return
	}
	transformDiagonalUnitary(x, y, archsimd.BroadcastFloat64x2(h11), archsimd.BroadcastFloat64x2(h22))
}

func rotmUnitaryDiagonalScalar(x, y []float64, h11, h22 float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = float64(vx*h11)+vy, -vx+float64(vy*h22)
	}
}

func transformUnitary(x, y []float64, h11, h12, h21, h22 archsimd.Float64x2) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	var i int
	for ; i+4 <= len(x); i += 4 {
		x0 := loadLevel1Float64x2(xData, i)
		x1 := loadLevel1Float64x2(xData, i+2)
		y0 := loadLevel1Float64x2(yData, i)
		y1 := loadLevel1Float64x2(yData, i+2)
		storeLevel1Float64x2(xData, i, x0.Mul(h11).Add(y0.Mul(h12)))
		storeLevel1Float64x2(xData, i+2, x1.Mul(h11).Add(y1.Mul(h12)))
		storeLevel1Float64x2(yData, i, x0.Mul(h21).Add(y0.Mul(h22)))
		storeLevel1Float64x2(yData, i+2, x1.Mul(h21).Add(y1.Mul(h22)))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx*h11.GetElem(0)+vy*h12.GetElem(0), vx*h21.GetElem(0)+vy*h22.GetElem(0)
	}
}

func transformOffDiagonalUnitary(x, y []float64, h12, h21 archsimd.Float64x2) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	var i int
	for ; i+4 <= len(x); i += 4 {
		x0 := loadLevel1Float64x2(xData, i)
		x1 := loadLevel1Float64x2(xData, i+2)
		y0 := loadLevel1Float64x2(yData, i)
		y1 := loadLevel1Float64x2(yData, i+2)
		storeLevel1Float64x2(xData, i, x0.Add(y0.Mul(h12)))
		storeLevel1Float64x2(xData, i+2, x1.Add(y1.Mul(h12)))
		storeLevel1Float64x2(yData, i, x0.Mul(h21).Add(y0))
		storeLevel1Float64x2(yData, i+2, x1.Mul(h21).Add(y1))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx+vy*h12.GetElem(0), vx*h21.GetElem(0)+vy
	}
}

func transformDiagonalUnitary(x, y []float64, h11, h22 archsimd.Float64x2) {
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	var i int
	for ; i+4 <= len(x); i += 4 {
		x0 := loadLevel1Float64x2(xData, i)
		x1 := loadLevel1Float64x2(xData, i+2)
		y0 := loadLevel1Float64x2(yData, i)
		y1 := loadLevel1Float64x2(yData, i+2)
		storeLevel1Float64x2(xData, i, x0.Mul(h11).Add(y0))
		storeLevel1Float64x2(xData, i+2, x1.Mul(h11).Add(y1))
		storeLevel1Float64x2(yData, i, x0.Neg().Add(y0.Mul(h22)))
		storeLevel1Float64x2(yData, i+2, x1.Neg().Add(y1.Mul(h22)))
	}
	for ; i < len(x); i++ {
		vx := x[i]
		vy := y[i]
		x[i], y[i] = vx*h11.GetElem(0)+vy, -vx+vy*h22.GetElem(0)
	}
}

func loadLevel1Float64x2(data unsafe.Pointer, i int) archsimd.Float64x2 {
	return archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(data, uintptr(i)*unsafe.Sizeof(float64(0)))))
}

func storeLevel1Float64x2(data unsafe.Pointer, i int, v archsimd.Float64x2) {
	v.StoreArray((*[2]float64)(unsafe.Add(data, uintptr(i)*unsafe.Sizeof(float64(0)))))
}

func level1SIMDCompatible(x, y []float64) bool {
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
