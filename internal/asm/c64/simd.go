// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import "simd"

func AxpyUnitarySIMD(alpha complex64, x, y []complex64) {
	AxpyUnitaryToSIMD(y, alpha, x, y)
}

func AxpyUnitaryToSIMD(dst []complex64, alpha complex64, x, y []complex64) {
	width := simd.BroadcastFloat32s(0).Len()
	var xr, xi, yr, yi, rr, ri [64]float32
	var i int
	for ; i+width <= len(x); i += width {
		portableLoadComplex64(x[i:], xr[:], xi[:], width)
		portableLoadComplex64(y[i:], yr[:], yi[:], width)
		complex64MulAdd(alpha, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		portableStoreComplex64(dst[i:], rr[:], ri[:], width)
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyIncSIMD(alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	AxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func AxpyIncToSIMD(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat32s(0).Len()
	var xr, xi, yr, yi, rr, ri [64]float32
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

func DotcUnitarySIMD(x, y []complex64) complex64 {
	return portableDotUnitarySIMD(x, y, true)
}

func DotuUnitarySIMD(x, y []complex64) complex64 {
	return portableDotUnitarySIMD(x, y, false)
}

func portableDotUnitarySIMD(x, y []complex64, conjugate bool) complex64 {
	width := simd.BroadcastFloat32s(0).Len()
	realAcc := simd.BroadcastFloat32s(0)
	imagAcc := simd.BroadcastFloat32s(0)
	var xr, xi, yr, yi [64]float32
	var i int
	for ; i+width <= len(x); i += width {
		portableLoadComplex64(x[i:], xr[:], xi[:], width)
		portableLoadComplex64(y[i:], yr[:], yi[:], width)
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
	sum := complex(portableReduceF32(realAcc), portableReduceF32(imagAcc))
	for ; i < len(x); i++ {
		if conjugate {
			sum += conj64(x[i]) * y[i]
		} else {
			sum += x[i] * y[i]
		}
	}
	return sum
}

func DotcIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, true)
}

func DotuIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, false)
}

func portableDotIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr, conjugate bool) complex64 {
	width := simd.BroadcastFloat32s(0).Len()
	realAcc := simd.BroadcastFloat32s(0)
	imagAcc := simd.BroadcastFloat32s(0)
	var xr, xi, yr, yi [64]float32
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
	sum := complex(portableReduceF32(realAcc), portableReduceF32(imagAcc))
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

func portableLoadComplex64(src []complex64, realPart, imagPart []float32, width int) {
	for i, value := range src[:width] {
		realPart[i], imagPart[i] = real(value), imag(value)
	}
}

func portableStoreComplex64(dst []complex64, realPart, imagPart []float32, width int) {
	for i := 0; i < width; i++ {
		dst[i] = complex(realPart[i], imagPart[i])
	}
}

func conj64(value complex64) complex64 {
	return complex(real(value), -imag(value))
}

func portableReduceF32(value simd.Float32s) float32 {
	var lanes [64]float32
	width := value.Len()
	value.Store(lanes[:])
	var sum float32
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}
