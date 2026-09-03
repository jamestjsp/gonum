// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && !safe && !noasm && !gccgo

package simdbench

import "simd"

func C128AxpyUnitarySIMD(alpha complex128, x, y []complex128) {
	C128AxpyUnitaryToSIMD(y, alpha, x, y)
}

func C128AxpyUnitaryToSIMD(dst []complex128, alpha complex128, x, y []complex128) {
	width := simd.BroadcastFloat64s(0).Len()
	var xr, xi, yr, yi, rr, ri [8]float64
	var i int
	for ; i+width <= len(x); i += width {
		loadComplex128(x[i:], xr[:], xi[:], width)
		loadComplex128(y[i:], yr[:], yi[:], width)
		complex128MulAdd(alpha, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		storeComplex128(dst[i:], rr[:], ri[:], width)
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func C128AxpyIncSIMD(alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	C128AxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func C128AxpyIncToSIMD(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	var xr, xi, yr, yi, rr, ri [8]float64
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xr[lane], xi[lane] = real(x[ix]), imag(x[ix])
			yr[lane], yi[lane] = real(y[iy]), imag(y[iy])
			ix += incX
			iy += incY
		}
		complex128MulAdd(alpha, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		for lane := 0; lane < width; lane++ {
			dst[idst] = complex(rr[lane], ri[lane])
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

func C128DscalUnitarySIMD(alpha float64, x []complex128) {
	c128DscaleSIMD(alpha, x, uintptr(len(x)), 1)
}

func C128DscalIncSIMD(alpha float64, x []complex128, n, inc uintptr) {
	c128DscaleSIMD(alpha, x, n, inc)
}

func C128ScalUnitarySIMD(alpha complex128, x []complex128) {
	c128ScaleSIMD(alpha, x, uintptr(len(x)), 1)
}

func C128ScalIncSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	c128ScaleSIMD(alpha, x, n, inc)
}

func c128DscaleSIMD(alpha float64, x []complex128, n, inc uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	a := simd.BroadcastFloat64s(alpha)
	var xr, xi [8]float64
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		start := index
		for lane := 0; lane < width; lane++ {
			xr[lane], xi[lane] = real(x[index]), imag(x[index])
			index += inc
		}
		simd.LoadFloat64s(xr[:]).Mul(a).Store(xr[:])
		simd.LoadFloat64s(xi[:]).Mul(a).Store(xi[:])
		write := start
		for lane := 0; lane < width; lane++ {
			x[write] = complex(xr[lane], xi[lane])
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

func c128ScaleSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	var xr, xi, zero, rr, ri [8]float64
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		start := index
		for lane := 0; lane < width; lane++ {
			xr[lane], xi[lane] = real(x[index]), imag(x[index])
			index += inc
		}
		complex128MulAdd(alpha, xr[:], xi[:], zero[:], zero[:], rr[:], ri[:])
		write := start
		for lane := 0; lane < width; lane++ {
			x[write] = complex(rr[lane], ri[lane])
			write += inc
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		x[index] *= alpha
		index += inc
	}
}

func C128DotcUnitarySIMD(x, y []complex128) complex128 {
	return c128DotUnitarySIMD(x, y, true)
}

func C128DotuUnitarySIMD(x, y []complex128) complex128 {
	return c128DotUnitarySIMD(x, y, false)
}

func c128DotUnitarySIMD(x, y []complex128, conjugate bool) complex128 {
	width := simd.BroadcastFloat64s(0).Len()
	realAcc := simd.BroadcastFloat64s(0)
	imagAcc := simd.BroadcastFloat64s(0)
	var xr, xi, yr, yi [8]float64
	var i int
	for ; i+width <= len(x); i += width {
		loadComplex128(x[i:], xr[:], xi[:], width)
		loadComplex128(y[i:], yr[:], yi[:], width)
		realAcc, imagAcc = complex128DotBlock(xr[:], xi[:], yr[:], yi[:], realAcc, imagAcc, conjugate)
	}
	sum := complex(reduceF64(realAcc), reduceF64(imagAcc))
	for ; i < len(x); i++ {
		if conjugate {
			sum += conj128(x[i]) * y[i]
		} else {
			sum += x[i] * y[i]
		}
	}
	return sum
}

func C128DotcIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
	return c128DotIncSIMD(x, y, n, incX, incY, ix, iy, true)
}

func C128DotuIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
	return c128DotIncSIMD(x, y, n, incX, incY, ix, iy, false)
}

func c128DotIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr, conjugate bool) complex128 {
	width := simd.BroadcastFloat64s(0).Len()
	realAcc := simd.BroadcastFloat64s(0)
	imagAcc := simd.BroadcastFloat64s(0)
	var xr, xi, yr, yi [8]float64
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xr[lane], xi[lane] = real(x[ix]), imag(x[ix])
			yr[lane], yi[lane] = real(y[iy]), imag(y[iy])
			ix += incX
			iy += incY
		}
		realAcc, imagAcc = complex128DotBlock(xr[:], xi[:], yr[:], yi[:], realAcc, imagAcc, conjugate)
		remaining -= width
	}
	sum := complex(reduceF64(realAcc), reduceF64(imagAcc))
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

func complex128DotBlock(xr, xi, yr, yi []float64, realAcc, imagAcc simd.Float64s, conjugate bool) (simd.Float64s, simd.Float64s) {
	xrv, xiv := simd.LoadFloat64s(xr), simd.LoadFloat64s(xi)
	yrv, yiv := simd.LoadFloat64s(yr), simd.LoadFloat64s(yi)
	if conjugate {
		return xrv.Mul(yrv).Add(xiv.Mul(yiv)).Add(realAcc), xrv.Mul(yiv).Sub(xiv.Mul(yrv)).Add(imagAcc)
	}
	return xrv.Mul(yrv).Sub(xiv.Mul(yiv)).Add(realAcc), xrv.Mul(yiv).Add(xiv.Mul(yrv)).Add(imagAcc)
}

func complex128MulAdd(alpha complex128, xr, xi, yr, yi, rr, ri []float64) {
	ar := simd.BroadcastFloat64s(real(alpha))
	ai := simd.BroadcastFloat64s(imag(alpha))
	xrv, xiv := simd.LoadFloat64s(xr), simd.LoadFloat64s(xi)
	xrv.Mul(ar).Sub(xiv.Mul(ai)).Add(simd.LoadFloat64s(yr)).Store(rr)
	xrv.Mul(ai).Add(xiv.Mul(ar)).Add(simd.LoadFloat64s(yi)).Store(ri)
}

func loadComplex128(src []complex128, realPart, imagPart []float64, width int) {
	for i, value := range src[:width] {
		realPart[i], imagPart[i] = real(value), imag(value)
	}
}

func storeComplex128(dst []complex128, realPart, imagPart []float64, width int) {
	for i := 0; i < width; i++ {
		dst[i] = complex(realPart[i], imagPart[i])
	}
}

func conj128(value complex128) complex128 {
	return complex(real(value), -imag(value))
}
