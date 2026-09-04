// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"simd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha complex64, x, y []complex64) {
	AxpyUnitaryToSIMD(y, alpha, x, y)
}

func AxpyUnitaryToSIMD(dst []complex64, alpha complex64, x, y []complex64) {
	width := simd.BroadcastFloat32s(0).Len()
	ar := simd.BroadcastFloat32s(real(alpha))
	ai := simd.BroadcastFloat32s(imag(alpha))
	xr := make([]uint32, width)
	xi := make([]uint32, width)
	yr := make([]uint32, width)
	yi := make([]uint32, width)
	rr := make([]uint32, width)
	ri := make([]uint32, width)
	var i int
	for ; i+width <= len(x); i += width {
		portableLoadComplex64(x[i:], xr[:], xi[:], width)
		portableLoadComplex64(y[i:], yr[:], yi[:], width)
		complex64MulAdd(ar, ai, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
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
	ar := simd.BroadcastFloat32s(real(alpha))
	ai := simd.BroadcastFloat32s(imag(alpha))
	xr := make([]uint32, width)
	xi := make([]uint32, width)
	yr := make([]uint32, width)
	yi := make([]uint32, width)
	rr := make([]uint32, width)
	ri := make([]uint32, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xv := (*[2]uint32)(unsafe.Pointer(&x[ix]))
			xr[lane], xi[lane] = xv[0], xv[1]
			yv := (*[2]uint32)(unsafe.Pointer(&y[iy]))
			yr[lane], yi[lane] = yv[0], yv[1]
			ix += incX
			iy += incY
		}
		complex64MulAdd(ar, ai, xr[:], xi[:], yr[:], yi[:], rr[:], ri[:])
		for lane := 0; lane < width; lane++ {
			value := (*[2]uint32)(unsafe.Pointer(&dst[idst]))
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
	xr := make([]uint32, width)
	xi := make([]uint32, width)
	yr := make([]uint32, width)
	yi := make([]uint32, width)
	y = y[:len(x):len(x)]
	for len(x) >= width {
		portableLoadComplex64(x[:width], xr[:], xi[:], width)
		portableLoadComplex64(y[:width], yr[:], yi[:], width)
		xrv, xiv := simd.LoadUint32s(xr[:]).BitsToFloat32(), simd.LoadUint32s(xi[:]).BitsToFloat32()
		yrv, yiv := simd.LoadUint32s(yr[:]).BitsToFloat32(), simd.LoadUint32s(yi[:]).BitsToFloat32()
		if conjugate {
			realAcc = xrv.Mul(yrv).Add(xiv.Mul(yiv)).Add(realAcc)
			imagAcc = xrv.Mul(yiv).Sub(xiv.Mul(yrv)).Add(imagAcc)
		} else {
			realAcc = xrv.Mul(yrv).Sub(xiv.Mul(yiv)).Add(realAcc)
			imagAcc = xrv.Mul(yiv).Add(xiv.Mul(yrv)).Add(imagAcc)
		}
		x, y = x[width:], y[width:]
	}
	sum := complex(portableReduceF32(realAcc), portableReduceF32(imagAcc))
	for i, value := range x {
		if conjugate {
			sum += conj64(value) * y[i]
		} else {
			sum += value * y[i]
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
	xr := make([]uint32, width)
	xi := make([]uint32, width)
	yr := make([]uint32, width)
	yi := make([]uint32, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xv := (*[2]uint32)(unsafe.Pointer(&x[ix]))
			xr[lane], xi[lane] = xv[0], xv[1]
			yv := (*[2]uint32)(unsafe.Pointer(&y[iy]))
			yr[lane], yi[lane] = yv[0], yv[1]
			ix += incX
			iy += incY
		}
		xrv, xiv := simd.LoadUint32s(xr[:]).BitsToFloat32(), simd.LoadUint32s(xi[:]).BitsToFloat32()
		yrv, yiv := simd.LoadUint32s(yr[:]).BitsToFloat32(), simd.LoadUint32s(yi[:]).BitsToFloat32()
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

func complex64MulAdd(ar, ai simd.Float32s, xr, xi, yr, yi, rr, ri []uint32) {
	xrv, xiv := simd.LoadUint32s(xr).BitsToFloat32(), simd.LoadUint32s(xi).BitsToFloat32()
	xrv.Mul(ar).Sub(xiv.Mul(ai)).Add(simd.LoadUint32s(yr).BitsToFloat32()).ToBits().Store(rr)
	xrv.Mul(ai).Add(xiv.Mul(ar)).Add(simd.LoadUint32s(yi).BitsToFloat32()).ToBits().Store(ri)
}

// Integer lane transfers avoid legacy SSE moves inside AMD64 AVX loops.
func portableLoadComplex64(src []complex64, realPart, imagPart []uint32, width int) {
	for i := range src[:width] {
		value := (*[2]uint32)(unsafe.Pointer(&src[i]))
		realPart[i], imagPart[i] = value[0], value[1]
	}
}

func portableStoreComplex64(dst []complex64, realPart, imagPart []uint32, width int) {
	for i := 0; i < width; i++ {
		value := (*[2]uint32)(unsafe.Pointer(&dst[i]))
		value[0], value[1] = realPart[i], imagPart[i]
	}
}

func conj64(value complex64) complex64 {
	return complex(real(value), -imag(value))
}

func portableReduceF32(value simd.Float32s) float32 {
	width := value.Len()
	lanes := make([]float32, width)
	value.Store(lanes[:])
	var sum float32
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}
