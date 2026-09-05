// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"unsafe"
)

// Retain the staged vector fallback for architectures without measured native
// strided kernels. AMD64 uses direct scalar arithmetic or masked stride-two
// vectors to avoid pack/scatter scratch traffic.
func axpyIncPortableSIMD(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
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

func axpyIncToPortableSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
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

func dotIncPortableSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
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
	sum := reduceF32Portable(acc)
	for ; remaining > 0; remaining-- {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}
