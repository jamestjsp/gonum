// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && !safe && !noasm && !gccgo

package simdbench

import "simd"

func C64AxpyUnitarySIMD(alpha complex64, x, y []complex64) {
	C64AxpyUnitaryToSIMD(y, alpha, x, y)
}

func C64AxpyUnitaryToSIMD(dst []complex64, alpha complex64, x, y []complex64) {
	width := simd.BroadcastFloat32s(0).Len()
	var xr, xi, yr, yi, rr, ri [16]float32
	var i int
	for ; i+width <= len(x); i += width {
		loadComplex64(x[i:], xr[:], xi[:], width)
		loadComplex64(y[i:], yr[:], yi[:], width)
		complex64MulAdd(alpha, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		storeComplex64(dst[i:], rr[:], ri[:], width)
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func C64AxpyIncSIMD(alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	C64AxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func C64AxpyIncToSIMD(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat32s(0).Len()
	var xr, xi, yr, yi, rr, ri [16]float32
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xr[lane], xi[lane] = real(x[ix]), imag(x[ix])
			yr[lane], yi[lane] = real(y[iy]), imag(y[iy])
			ix += incX
			iy += incY
		}
		complex64MulAdd(alpha, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
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

func C64DotcUnitarySIMD(x, y []complex64) complex64 {
	return c64DotUnitarySIMD(x, y, true)
}

func C64DotuUnitarySIMD(x, y []complex64) complex64 {
	return c64DotUnitarySIMD(x, y, false)
}

func c64DotUnitarySIMD(x, y []complex64, conjugate bool) complex64 {
	width := simd.BroadcastFloat32s(0).Len()
	realAcc := simd.BroadcastFloat32s(0)
	imagAcc := simd.BroadcastFloat32s(0)
	var xr, xi, yr, yi [16]float32
	var i int
	for ; i+width <= len(x); i += width {
		loadComplex64(x[i:], xr[:], xi[:], width)
		loadComplex64(y[i:], yr[:], yi[:], width)
		xrv, xiv := simd.LoadFloat32s(xr[:]), simd.LoadFloat32s(xi[:])
		yrv, yiv := simd.LoadFloat32s(yr[:]), simd.LoadFloat32s(yi[:])
		if conjugate {
			realAcc = xrv.Mul(yrv).Add(xiv.Mul(yiv)).Add(realAcc)
			imagAcc = xrv.Mul(yiv).Sub(xiv.Mul(yrv)).Add(imagAcc)
		} else {
			realAcc = xrv.Mul(yrv).Sub(xiv.Mul(yiv)).Add(realAcc)
			imagAcc = xrv.Mul(yiv).Add(xiv.Mul(yrv)).Add(imagAcc)
		}
	}
	sum := complex(reduceF32(realAcc), reduceF32(imagAcc))
	for ; i < len(x); i++ {
		if conjugate {
			sum += conj64(x[i]) * y[i]
		} else {
			sum += x[i] * y[i]
		}
	}
	return sum
}

func C64DotcIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	return c64DotIncSIMD(x, y, n, incX, incY, ix, iy, true)
}

func C64DotuIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	return c64DotIncSIMD(x, y, n, incX, incY, ix, iy, false)
}

func c64DotIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr, conjugate bool) complex64 {
	width := simd.BroadcastFloat32s(0).Len()
	realAcc := simd.BroadcastFloat32s(0)
	imagAcc := simd.BroadcastFloat32s(0)
	var xr, xi, yr, yi [16]float32
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xr[lane], xi[lane] = real(x[ix]), imag(x[ix])
			yr[lane], yi[lane] = real(y[iy]), imag(y[iy])
			ix += incX
			iy += incY
		}
		xrv, xiv := simd.LoadFloat32s(xr[:]), simd.LoadFloat32s(xi[:])
		yrv, yiv := simd.LoadFloat32s(yr[:]), simd.LoadFloat32s(yi[:])
		if conjugate {
			realAcc = xrv.Mul(yrv).Add(xiv.Mul(yiv)).Add(realAcc)
			imagAcc = xrv.Mul(yiv).Sub(xiv.Mul(yrv)).Add(imagAcc)
		} else {
			realAcc = xrv.Mul(yrv).Sub(xiv.Mul(yiv)).Add(realAcc)
			imagAcc = xrv.Mul(yiv).Add(xiv.Mul(yrv)).Add(imagAcc)
		}
		remaining -= width
	}
	sum := complex(reduceF32(realAcc), reduceF32(imagAcc))
	for ; remaining > 0; remaining-- {
		if conjugate {
			sum += conj64(x[ix]) * y[iy]
		} else {
			sum += x[ix] * y[iy]
		}
		ix += incX
		iy += incY
	}
	return sum
}

func complex64MulAdd(alpha complex64, xr, xi, yr, yi, rr, ri []float32) {
	ar := simd.BroadcastFloat32s(real(alpha))
	ai := simd.BroadcastFloat32s(imag(alpha))
	xrv, xiv := simd.LoadFloat32s(xr), simd.LoadFloat32s(xi)
	xrv.Mul(ar).Sub(xiv.Mul(ai)).Add(simd.LoadFloat32s(yr)).Store(rr)
	xrv.Mul(ai).Add(xiv.Mul(ar)).Add(simd.LoadFloat32s(yi)).Store(ri)
}

func loadComplex64(src []complex64, realPart, imagPart []float32, width int) {
	for i, value := range src[:width] {
		realPart[i], imagPart[i] = real(value), imag(value)
	}
}

func storeComplex64(dst []complex64, realPart, imagPart []float32, width int) {
	for i := 0; i < width; i++ {
		dst[i] = complex(realPart[i], imagPart[i])
	}
}

func conj64(value complex64) complex64 {
	return complex(real(value), -imag(value))
}
