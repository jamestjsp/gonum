// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"simd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha complex128, x, y []complex128) {
	AxpyUnitaryToSIMD(y, alpha, x, y)
}

func AxpyUnitaryToSIMD(dst []complex128, alpha complex128, x, y []complex128) {
	width := simd.BroadcastFloat64s(0).Len()
	ar := simd.BroadcastFloat64s(real(alpha))
	ai := simd.BroadcastFloat64s(imag(alpha))
	xr := make([]uint64, width)
	xi := make([]uint64, width)
	yr := make([]uint64, width)
	yi := make([]uint64, width)
	rr := make([]uint64, width)
	ri := make([]uint64, width)
	var i int
	for ; i+width <= len(x); i += width {
		portableLoadComplex128(x[i:], xr[:], xi[:], width)
		portableLoadComplex128(y[i:], yr[:], yi[:], width)
		complex128MulAdd(ar, ai, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		portableStoreComplex128(dst[i:], rr[:], ri[:], width)
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyIncSIMD(alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	AxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func AxpyIncToSIMD(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	ar := simd.BroadcastFloat64s(real(alpha))
	ai := simd.BroadcastFloat64s(imag(alpha))
	xr := make([]uint64, width)
	xi := make([]uint64, width)
	yr := make([]uint64, width)
	yi := make([]uint64, width)
	rr := make([]uint64, width)
	ri := make([]uint64, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xv := (*[2]uint64)(unsafe.Pointer(&x[ix]))
			xr[lane], xi[lane] = xv[0], xv[1]
			yv := (*[2]uint64)(unsafe.Pointer(&y[iy]))
			yr[lane], yi[lane] = yv[0], yv[1]
			ix += incX
			iy += incY
		}
		complex128MulAdd(ar, ai, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		for lane := 0; lane < width; lane++ {
			value := (*[2]uint64)(unsafe.Pointer(&dst[idst]))
			value[0], value[1] = rr[lane], ri[lane]
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

func DscalUnitarySIMD(alpha float64, x []complex128) {
	portableDscaleSIMD(alpha, x, uintptr(len(x)), 1)
}

func DscalIncSIMD(alpha float64, x []complex128, n, inc uintptr) {
	portableDscaleSIMD(alpha, x, n, inc)
}

func ScalUnitarySIMD(alpha complex128, x []complex128) {
	portableScaleSIMD(alpha, x, uintptr(len(x)), 1)
}

func ScalIncSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	portableScaleSIMD(alpha, x, n, inc)
}

func portableDscaleSIMD(alpha float64, x []complex128, n, inc uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	a := simd.BroadcastFloat64s(alpha)
	xr := make([]uint64, width)
	xi := make([]uint64, width)
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		start := index
		for lane := 0; lane < width; lane++ {
			xv := (*[2]uint64)(unsafe.Pointer(&x[index]))
			xr[lane], xi[lane] = xv[0], xv[1]
			index += inc
		}
		simd.LoadUint64s(xr[:]).BitsToFloat64().Mul(a).ToBits().Store(xr[:])
		simd.LoadUint64s(xi[:]).BitsToFloat64().Mul(a).ToBits().Store(xi[:])
		write := start
		for lane := 0; lane < width; lane++ {
			value := (*[2]uint64)(unsafe.Pointer(&x[write]))
			value[0], value[1] = xr[lane], xi[lane]
			write += inc
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		value := x[index]
		x[index] = complex(alpha*real(value), alpha*imag(value))
		index += inc
	}
}

func portableScaleSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	ar := simd.BroadcastFloat64s(real(alpha))
	ai := simd.BroadcastFloat64s(imag(alpha))
	xr := make([]uint64, width)
	xi := make([]uint64, width)
	zero := make([]uint64, width)
	rr := make([]uint64, width)
	ri := make([]uint64, width)
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		start := index
		for lane := 0; lane < width; lane++ {
			xv := (*[2]uint64)(unsafe.Pointer(&x[index]))
			xr[lane], xi[lane] = xv[0], xv[1]
			index += inc
		}
		complex128MulAdd(ar, ai, xr[:], xi[:], zero[:], zero[:], rr[:], ri[:])
		write := start
		for lane := 0; lane < width; lane++ {
			value := (*[2]uint64)(unsafe.Pointer(&x[write]))
			value[0], value[1] = rr[lane], ri[lane]
			write += inc
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		x[index] *= alpha
		index += inc
	}
}

func DotcUnitarySIMD(x, y []complex128) complex128 {
	return portableDotUnitarySIMD(x, y, true)
}

func DotuUnitarySIMD(x, y []complex128) complex128 {
	return portableDotUnitarySIMD(x, y, false)
}

func portableDotUnitarySIMD(x, y []complex128, conjugate bool) complex128 {
	width := simd.BroadcastFloat64s(0).Len()
	realAcc := simd.BroadcastFloat64s(0)
	imagAcc := simd.BroadcastFloat64s(0)
	xr := make([]uint64, width)
	xi := make([]uint64, width)
	yr := make([]uint64, width)
	yi := make([]uint64, width)
	y = y[:len(x):len(x)]
	for len(x) >= width {
		portableLoadComplex128(x[:width], xr[:], xi[:], width)
		portableLoadComplex128(y[:width], yr[:], yi[:], width)
		realAcc, imagAcc = complex128DotBlock(xr[:], xi[:], yr[:], yi[:], realAcc, imagAcc, conjugate)
		x, y = x[width:], y[width:]
	}
	sum := complex(portableReduceF64(realAcc), portableReduceF64(imagAcc))
	for i, value := range x {
		if conjugate {
			sum += conj128(value) * y[i]
		} else {
			sum += value * y[i]
		}
	}
	return sum
}

func DotcIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, true)
}

func DotuIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, false)
}

func portableDotIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr, conjugate bool) complex128 {
	width := simd.BroadcastFloat64s(0).Len()
	realAcc := simd.BroadcastFloat64s(0)
	imagAcc := simd.BroadcastFloat64s(0)
	xr := make([]uint64, width)
	xi := make([]uint64, width)
	yr := make([]uint64, width)
	yi := make([]uint64, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xv := (*[2]uint64)(unsafe.Pointer(&x[ix]))
			xr[lane], xi[lane] = xv[0], xv[1]
			yv := (*[2]uint64)(unsafe.Pointer(&y[iy]))
			yr[lane], yi[lane] = yv[0], yv[1]
			ix += incX
			iy += incY
		}
		realAcc, imagAcc = complex128DotBlock(xr[:], xi[:], yr[:], yi[:], realAcc, imagAcc, conjugate)
		remaining -= width
	}
	sum := complex(portableReduceF64(realAcc), portableReduceF64(imagAcc))
	for ; remaining > 0; remaining-- {
		if conjugate {
			sum += conj128(x[ix]) * y[iy]
		} else {
			sum += x[ix] * y[iy]
		}
		ix += incX
		iy += incY
	}
	return sum
}

func complex128DotBlock(xr, xi, yr, yi []uint64, realAcc, imagAcc simd.Float64s, conjugate bool) (simd.Float64s, simd.Float64s) {
	xrv, xiv := simd.LoadUint64s(xr).BitsToFloat64(), simd.LoadUint64s(xi).BitsToFloat64()
	yrv, yiv := simd.LoadUint64s(yr).BitsToFloat64(), simd.LoadUint64s(yi).BitsToFloat64()
	if conjugate {
		return xrv.Mul(yrv).Add(xiv.Mul(yiv)).Add(realAcc), xrv.Mul(yiv).Sub(xiv.Mul(yrv)).Add(imagAcc)
	}
	return xrv.Mul(yrv).Sub(xiv.Mul(yiv)).Add(realAcc), xrv.Mul(yiv).Add(xiv.Mul(yrv)).Add(imagAcc)
}

func complex128MulAdd(ar, ai simd.Float64s, xr, xi, yr, yi, rr, ri []uint64) {
	xrv, xiv := simd.LoadUint64s(xr).BitsToFloat64(), simd.LoadUint64s(xi).BitsToFloat64()
	xrv.Mul(ar).Sub(xiv.Mul(ai)).Add(simd.LoadUint64s(yr).BitsToFloat64()).ToBits().Store(rr)
	xrv.Mul(ai).Add(xiv.Mul(ar)).Add(simd.LoadUint64s(yi).BitsToFloat64()).ToBits().Store(ri)
}

// Integer lane transfers avoid legacy SSE moves inside AMD64 AVX loops.
func portableLoadComplex128(src []complex128, realPart, imagPart []uint64, width int) {
	for i := range src[:width] {
		value := (*[2]uint64)(unsafe.Pointer(&src[i]))
		realPart[i], imagPart[i] = value[0], value[1]
	}
}

func portableStoreComplex128(dst []complex128, realPart, imagPart []uint64, width int) {
	for i := 0; i < width; i++ {
		value := (*[2]uint64)(unsafe.Pointer(&dst[i]))
		value[0], value[1] = realPart[i], imagPart[i]
	}
}

func conj128(value complex128) complex128 {
	return complex(real(value), -imag(value))
}

func portableReduceF64(value simd.Float64s) float64 {
	width := value.Len()
	lanes := make([]float64, width)
	value.Store(lanes[:])
	var sum float64
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}
