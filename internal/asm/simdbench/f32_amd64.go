// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && !safe && !noasm && !gccgo

package simdbench

import "simd"

func F32AxpyUnitarySIMD(alpha float32, x, y []float32) {
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

func F32AxpyUnitaryToSIMD(dst []float32, alpha float32, x, y []float32) {
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

func F32AxpyIncSIMD(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat32s(0).Len()
	a := simd.BroadcastFloat32s(alpha)
	var xb, yb [16]float32
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

func F32AxpyIncToSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat32s(0).Len()
	a := simd.BroadcastFloat32s(alpha)
	var xb, yb, out [16]float32
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

func F32DotUnitarySIMD(x, y []float32) float32 {
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

func F32DotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	acc := simd.BroadcastFloat32s(0)
	width := acc.Len()
	var xb, yb [16]float32
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

func F32DdotUnitarySIMD(x, y []float32) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var xb, yb [8]float64
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

func F32DdotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var xb, yb [8]float64
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

func F32SumSIMD(x []float32) float32 {
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
	var lanes [16]float32
	width := value.Len()
	value.Store(lanes[:])
	var sum float32
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}

func F32GerSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) {
	var ix, iy uintptr
	if int(incX) < 0 {
		ix = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		iy = uintptr(-int(n-1) * int(incY))
	}
	for row := uintptr(0); row < m; row++ {
		F32AxpyIncSIMD(alpha*x[ix], y, a[row*lda:row*lda+n], n, incY, 1, iy, 0)
		ix += incX
	}
}
