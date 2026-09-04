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
	y = y[:len(x):len(x)]
	for len(x) >= width {
		simd.LoadFloat32s(x[:width]).Mul(a).Add(simd.LoadFloat32s(y[:width])).Store(y[:width])
		x, y = x[width:], y[width:]
	}
	for i, value := range x {
		y[i] += alpha * value
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
	y, dst = y[:len(x):len(x)], dst[:len(x):len(x)]
	for len(x) >= width {
		simd.LoadFloat32s(x[:width]).Mul(a).Add(simd.LoadFloat32s(y[:width])).Store(dst[:width])
		x, y, dst = x[width:], y[width:], dst[width:]
	}
	for i, value := range x {
		dst[i] = alpha*value + y[i]
	}
}

func AxpyIncSIMD(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		AxpyUnitarySIMD(alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incX == 0 || incY == 0 || !simdSlicesCompatible(x, y) || unsafe.SliceData(x) == unsafe.SliceData(y) {
		for ; n > 0; n-- {
			y[iy] += alpha * x[ix]
			ix += incX
			iy += incY
		}
		return
	}
	width := simd.BroadcastFloat32s(0).Len()
	a := simd.BroadcastFloat32s(alpha)
	// Integer staging avoids legacy SSE loads between AVX operations on amd64.
	xb, yb := make([]uint32, width), make([]uint32, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = *(*uint32)(unsafe.Pointer(&x[ix]))
			yb[lane] = *(*uint32)(unsafe.Pointer(&y[iy]))
			ix += incX
			iy += incY
		}
		simd.LoadUint32s(xb).BitsToFloat32().Mul(a).Add(simd.LoadUint32s(yb).BitsToFloat32()).ToBits().Store(yb)
		write := iy - uintptr(width)*incY
		for lane := 0; lane < width; lane++ {
			*(*uint32)(unsafe.Pointer(&y[write])) = yb[lane]
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
	if n == 0 {
		return
	}
	if incDst == 1 && incX == 1 && incY == 1 {
		AxpyUnitaryToSIMD(dst[idst:idst+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incDst == 0 || incX == 0 || incY == 0 || !simdSlicesCompatible(dst, x) || !simdSlicesCompatible(dst, y) || unsafe.SliceData(dst) == unsafe.SliceData(x) || unsafe.SliceData(dst) == unsafe.SliceData(y) {
		for ; n > 0; n-- {
			dst[idst] = alpha*x[ix] + y[iy]
			ix += incX
			iy += incY
			idst += incDst
		}
		return
	}
	width := simd.BroadcastFloat32s(0).Len()
	a := simd.BroadcastFloat32s(alpha)
	xb, yb, out := make([]uint32, width), make([]uint32, width), make([]uint32, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = *(*uint32)(unsafe.Pointer(&x[ix]))
			yb[lane] = *(*uint32)(unsafe.Pointer(&y[iy]))
			ix += incX
			iy += incY
		}
		simd.LoadUint32s(xb).BitsToFloat32().Mul(a).Add(simd.LoadUint32s(yb).BitsToFloat32()).ToBits().Store(out)
		for lane := 0; lane < width; lane++ {
			*(*uint32)(unsafe.Pointer(&dst[idst])) = out[lane]
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
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	y = y[:len(x):len(x)]
	for len(x) >= 4*width {
		xblock, yblock := x[:4*width], y[:4*width]
		acc = simd.LoadFloat32s(xblock[:width]).Mul(simd.LoadFloat32s(yblock[:width])).Add(acc)
		acc1 = simd.LoadFloat32s(xblock[width : 2*width]).Mul(simd.LoadFloat32s(yblock[width : 2*width])).Add(acc1)
		acc2 = simd.LoadFloat32s(xblock[2*width : 3*width]).Mul(simd.LoadFloat32s(yblock[2*width : 3*width])).Add(acc2)
		acc3 = simd.LoadFloat32s(xblock[3*width : 4*width]).Mul(simd.LoadFloat32s(yblock[3*width : 4*width])).Add(acc3)
		x, y = x[4*width:], y[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat32s(x[:width]).Mul(simd.LoadFloat32s(y[:width])).Add(acc)
		x, y = x[width:], y[width:]
	}
	sum := reduceF32(acc)
	for i, value := range x {
		sum += value * y[i]
	}
	return sum
}

func DotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return DotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	acc := simd.BroadcastFloat32s(0)
	width := acc.Len()
	xb, yb := make([]uint32, width), make([]uint32, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = *(*uint32)(unsafe.Pointer(&x[ix]))
			yb[lane] = *(*uint32)(unsafe.Pointer(&y[iy]))
			ix += incX
			iy += incY
		}
		acc = simd.LoadUint32s(xb).BitsToFloat32().Mul(simd.LoadUint32s(yb).BitsToFloat32()).Add(acc)
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
	xb, yb := make([]float64, width), make([]float64, width)
	y = y[:len(x):len(x)]
	for len(x) >= width {
		for lane, value := range x[:width] {
			xb[lane] = float64(value)
			yb[lane] = float64(y[:width][lane])
		}
		acc = simd.LoadFloat64s(xb[:]).Mul(simd.LoadFloat64s(yb[:])).Add(acc)
		x, y = x[width:], y[width:]
	}
	sum := reduceF64(acc)
	for i, value := range x {
		sum += float64(value) * float64(y[i])
	}
	return sum
}

func DdotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return DdotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	xb, yb := make([]float64, width), make([]float64, width)
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
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	for len(x) >= 4*width {
		acc = simd.LoadFloat32s(x[:width]).Add(acc)
		acc1 = simd.LoadFloat32s(x[width : 2*width]).Add(acc1)
		acc2 = simd.LoadFloat32s(x[2*width : 3*width]).Add(acc2)
		acc3 = simd.LoadFloat32s(x[3*width : 4*width]).Add(acc3)
		x = x[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat32s(x[:width]).Add(acc)
		x = x[width:]
	}
	sum := reduceF32(acc)
	for _, value := range x {
		sum += value
	}
	return sum
}

func reduceF32(value simd.Float32s) float32 {
	width := value.Len()
	lanes := make([]float32, width)
	value.Store(lanes[:])
	var sum float32
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}

func reduceF64(value simd.Float64s) float64 {
	width := value.Len()
	lanes := make([]float64, width)
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
