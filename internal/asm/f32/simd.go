// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha float32, x, y []float32) {
	if !simdSlicesCompatible(x, y[:len(x)]) {
		for i, value := range x {
			y[i] += alpha * value
		}
		return
	}
	a := simd.BroadcastFloat32s(alpha)
	width := a.Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat32s(x[i:]).Mul(a).Add(simd.LoadFloat32s(y[i:])).Store(y[i:])
	}
	for ; i < len(x); i++ {
		y[i] += alpha * x[i]
	}
}

func AxpyUnitaryToSIMD(dst []float32, alpha float32, x, y []float32) {
	if !simdSlicesCompatible(dst[:len(x)], x) || !simdSlicesCompatible(dst[:len(x)], y[:len(x)]) {
		for i, value := range x {
			dst[i] = alpha*value + y[i]
		}
		return
	}
	a := simd.BroadcastFloat32s(alpha)
	width := a.Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat32s(x[i:]).Mul(a).Add(simd.LoadFloat32s(y[i:])).Store(dst[i:])
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyIncSIMD(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat32s(0).Len()
	a := simd.BroadcastFloat32s(alpha)
	var xb, yb [64]float32
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = x[ix]
			yb[lane] = y[iy]
			ix += incX
			iy += incY
		}
		simd.LoadFloat32s(xb[:]).Mul(a).Add(simd.LoadFloat32s(yb[:])).Store(yb[:])
		write := iy - uintptr(width)*incY
		for lane := 0; lane < width; lane++ {
			y[write] = yb[lane]
			write += incY
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func AxpyIncToSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat32s(0).Len()
	a := simd.BroadcastFloat32s(alpha)
	var xb, yb, out [64]float32
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = x[ix]
			yb[lane] = y[iy]
			ix += incX
			iy += incY
		}
		simd.LoadFloat32s(xb[:]).Mul(a).Add(simd.LoadFloat32s(yb[:])).Store(out[:])
		for lane := 0; lane < width; lane++ {
			dst[idst] = out[lane]
			idst += incDst
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

func DotUnitarySIMD(x, y []float32) float32 {
	acc := simd.BroadcastFloat32s(0)
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		acc = simd.LoadFloat32s(x[i:]).Mul(simd.LoadFloat32s(y[i:])).Add(acc)
	}
	sum := reduceF32(acc)
	for ; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}

func DotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	acc := simd.BroadcastFloat32s(0)
	width := acc.Len()
	var xb, yb [64]float32
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = x[ix]
			yb[lane] = y[iy]
			ix += incX
			iy += incY
		}
		acc = simd.LoadFloat32s(xb[:]).Mul(simd.LoadFloat32s(yb[:])).Add(acc)
		remaining -= width
	}
	sum := reduceF32(acc)
	for ; remaining > 0; remaining-- {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func DdotUnitarySIMD(x, y []float32) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var xb, yb [32]float64
	var i int
	for ; i+width <= len(x); i += width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = float64(x[i+lane])
			yb[lane] = float64(y[i+lane])
		}
		acc = simd.LoadFloat64s(xb[:]).Mul(simd.LoadFloat64s(yb[:])).Add(acc)
	}
	sum := reduceF64(acc)
	for ; i < len(x); i++ {
		sum += float64(x[i]) * float64(y[i])
	}
	return sum
}

func DdotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var xb, yb [32]float64
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = float64(x[ix])
			yb[lane] = float64(y[iy])
			ix += incX
			iy += incY
		}
		acc = simd.LoadFloat64s(xb[:]).Mul(simd.LoadFloat64s(yb[:])).Add(acc)
		remaining -= width
	}
	sum := reduceF64(acc)
	for ; remaining > 0; remaining-- {
		sum += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
	}
	return sum
}

func SumSIMD(x []float32) float32 {
	acc := simd.BroadcastFloat32s(0)
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		acc = simd.LoadFloat32s(x[i:]).Add(acc)
	}
	sum := reduceF32(acc)
	for ; i < len(x); i++ {
		sum += x[i]
	}
	return sum
}

func reduceF32(value simd.Float32s) float32 {
	var lanes [64]float32
	width := value.Len()
	value.Store(lanes[:])
	var sum float32
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}

func reduceF64(value simd.Float64s) float64 {
	var lanes [32]float64
	width := value.Len()
	value.Store(lanes[:])
	var sum float64
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}

func GerSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) {
	var ix, iy uintptr
	if int(incX) < 0 {
		ix = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		iy = uintptr(-int(n-1) * int(incY))
	}
	for row := uintptr(0); row < m; row++ {
		AxpyIncSIMD(alpha*x[ix], y, a[row*lda:row*lda+n], n, incY, 1, iy, 0)
		ix += incX
	}
}

func simdSlicesCompatible(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if aStart == bStart {
		return true
	}
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aEnd <= bStart || bEnd <= aStart
}
