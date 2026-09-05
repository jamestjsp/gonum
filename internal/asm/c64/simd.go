// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha complex64, x, y []complex64) {
	AxpyUnitaryToSIMD(y, alpha, x, y)
}

func AxpyUnitaryToSIMD(dst []complex64, alpha complex64, x, y []complex64) {
	n := len(x)
	if n < 4 || !complexSlicesCompatibleSIMD(x, dst) || !complexSlicesCompatibleSIMD(y, dst) {
		for i, value := range x {
			dst[i] = alpha*value + y[i]
		}
		return
	}
	xf, yf, df := complexFloatsSIMD(x), complexFloatsSIMD(y[:n]), complexFloatsSIMD(dst[:n])
	width := simd.BroadcastFloat32s(0).Len()
	ar := simd.BroadcastFloat32s(real(alpha))
	// Negating even imaginary products implements the real subtraction without FMA.
	ai := simd.BroadcastFloat32s(imag(alpha)).ToBits().Xor(complexEvenSignSIMD()).BitsToFloat32()
	i := 0
	for ; i+width <= len(xf); i += width {
		xv, yv := simd.LoadFloat32s(xf[i:i+width]), simd.LoadFloat32s(yf[i:i+width])
		xv.Mul(ar).Add(complexSwapSIMD(xv).Mul(ai)).Add(yv).Store(df[i : i+width])
	}
	i /= 2
	if width >= 16 && n-i >= 4 && complexNativeSIMD() {
		i += complexAxpyTailNativeSIMD(dst[i:n], alpha, x[i:], y[i:n])
	}

	for ; i < n; i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyIncSIMD(alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	AxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func AxpyIncToSIMD(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 && incDst == 1 {
		AxpyUnitaryToSIMD(dst[idst:idst+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incX == 0 || incY == 0 || incDst == 0 || !complexIncrementsCompatibleSIMD(x, dst, incX, incDst, ix, idst) || !complexIncrementsCompatibleSIMD(y, dst, incY, incDst, iy, idst) {
		for ; n > 0; n-- {
			dst[idst] = alpha*x[ix] + y[iy]
			ix += incX
			iy += incY
			idst += incDst
		}
		return
	}

	if complexNativeSIMD() {
		complexAxpyIncNativeSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy)
		return
	}

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
	sum := interleavedDotUnitarySIMD(x, y, conjugate)
	if !math.IsNaN(float64(real(sum))) && !math.IsNaN(float64(imag(sum))) && !math.IsInf(float64(real(sum)), 0) && !math.IsInf(float64(imag(sum)), 0) {
		return sum
	}
	// Separate component sums can overflow before per-element cancellation.
	sum = 0
	for i, v := range x {
		if conjugate {
			v = conj64(v)
		}
		sum += v * y[i]
	}
	return sum
}

func interleavedDotUnitarySIMD(x, y []complex64, conjugate bool) complex64 {
	if len(x) < 4 {
		var sum complex64
		for i, value := range x {
			if conjugate {
				value = conj64(value)
			}
			sum += value * y[i]
		}
		return sum
	}
	// At 256 bits, use the wide loop from 32 elements to avoid short-kernel
	// setup overhead. Preserve the wider cutoff at the other native widths.
	shortLimit := 64
	if simd.BroadcastFloat32s(0).Len() == 8 {
		shortLimit = 32
	}
	if len(x) < shortLimit && complexNativeSIMD() {
		sum := complexDotShortNativeSIMD(x, y, conjugate)
		if !math.IsNaN(float64(real(sum))) && !math.IsNaN(float64(imag(sum))) && !math.IsInf(float64(real(sum)), 0) && !math.IsInf(float64(imag(sum)), 0) {
			return sum
		}
		// A different grouping may avoid intermediate overflow. Retry the
		// established vector algorithm before the wrapper recovers sequentially.
	}

	xf, yf := complexFloatsSIMD(x), complexFloatsSIMD(y[:len(x)])
	width := simd.BroadcastFloat32s(0).Len()
	r0, r1 := simd.BroadcastFloat32s(0), simd.BroadcastFloat32s(0)
	r2, r3 := simd.BroadcastFloat32s(0), simd.BroadcastFloat32s(0)
	i0, i1 := simd.BroadcastFloat32s(0), simd.BroadcastFloat32s(0)
	i2, i3 := simd.BroadcastFloat32s(0), simd.BroadcastFloat32s(0)
	i := 0
	for ; i+4*width <= len(xf); i += 4 * width {
		xv0, yv0 := simd.LoadFloat32s(xf[i:i+width]), simd.LoadFloat32s(yf[i:i+width])
		xv1, yv1 := simd.LoadFloat32s(xf[i+width:i+2*width]), simd.LoadFloat32s(yf[i+width:i+2*width])
		xv2, yv2 := simd.LoadFloat32s(xf[i+2*width:i+3*width]), simd.LoadFloat32s(yf[i+2*width:i+3*width])
		xv3, yv3 := simd.LoadFloat32s(xf[i+3*width:i+4*width]), simd.LoadFloat32s(yf[i+3*width:i+4*width])
		r0 = r0.Add(xv0.Mul(yv0))
		i0 = i0.Add(xv0.Mul(complexSwapSIMD(yv0)))
		r1 = r1.Add(xv1.Mul(yv1))
		i1 = i1.Add(xv1.Mul(complexSwapSIMD(yv1)))
		r2 = r2.Add(xv2.Mul(yv2))
		i2 = i2.Add(xv2.Mul(complexSwapSIMD(yv2)))
		r3 = r3.Add(xv3.Mul(yv3))
		i3 = i3.Add(xv3.Mul(complexSwapSIMD(yv3)))
	}
	r0, i0 = r0.Add(r1).Add(r2).Add(r3), i0.Add(i1).Add(i2).Add(i3)
	for ; i+width <= len(xf); i += width {
		xv, yv := simd.LoadFloat32s(xf[i:i+width]), simd.LoadFloat32s(yf[i:i+width])
		r0 = r0.Add(xv.Mul(yv))
		i0 = i0.Add(xv.Mul(complexSwapSIMD(yv)))
	}
	r, im := make([]float32, width), make([]float32, width)
	r0.Store(r)
	i0.Store(im)
	var realSum, imagSum float32
	for lane := 0; lane < width; lane += 2 {
		if conjugate {
			realSum += r[lane] + r[lane+1]
			imagSum += im[lane] - im[lane+1]
		} else {
			realSum += r[lane] - r[lane+1]
			imagSum += im[lane] + im[lane+1]
		}
	}
	sum := complex(realSum, imagSum)
	for i /= 2; i < len(x); i++ {
		value := x[i]
		if conjugate {
			value = conj64(value)
		}
		sum += value * y[i]
	}
	return sum
}

func DotcIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return portableDotUnitarySIMD(x[ix:ix+n], y[iy:iy+n], true)
	}
	if complexNativeSIMD() {
		return complexDotIncNativeSIMD(x, y, n, incX, incY, ix, iy, true)
	}
	return portableDotIncSIMD(x, y, n, incX, incY, ix, iy, true)
}

func DotuIncSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr) complex64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return portableDotUnitarySIMD(x[ix:ix+n], y[iy:iy+n], false)
	}
	if complexNativeSIMD() {
		return complexDotIncNativeSIMD(x, y, n, incX, incY, ix, iy, false)
	}
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
func complexFloatsSIMD(x []complex64) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(unsafe.SliceData(x))), 2*len(x))
}

func complexSlicesCompatibleSIMD(x, y []complex64) bool {
	if len(x) == 0 || len(y) == 0 {
		return true
	}
	xp, yp := uintptr(unsafe.Pointer(unsafe.SliceData(x))), uintptr(unsafe.Pointer(unsafe.SliceData(y)))
	const size = unsafe.Sizeof(complex64(0))
	return xp == yp || xp+uintptr(len(x))*size <= yp || yp+uintptr(len(y))*size <= xp
}

func conj64(value complex64) complex64 {
	return complex(real(value), -imag(value))
}

// Rotating each complex number's 64-bit representation swaps its 32-bit components.
func complexSwapSIMD(x simd.Float32s) simd.Float32s {
	return x.ToBits().ReshapeToUint64s().RotateAllLeft(32).ReshapeToUint32s().BitsToFloat32()
}

func complexEvenSignSIMD() simd.Uint32s {
	return simd.BroadcastUint64s(1 << 31).ReshapeToUint32s()
}

func complex64MulAdd(ar, ai simd.Float32s, xr, xi, yr, yi, rr, ri []uint32) {
	xrv, xiv := simd.LoadUint32s(xr).BitsToFloat32(), simd.LoadUint32s(xi).BitsToFloat32()
	xrv.Mul(ar).Sub(xiv.Mul(ai)).Add(simd.LoadUint32s(yr).BitsToFloat32()).ToBits().Store(rr)
	xrv.Mul(ai).Add(xiv.Mul(ar)).Add(simd.LoadUint32s(yi).BitsToFloat32()).ToBits().Store(ri)
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

// Equal base addresses are safe for vector staging only when the index streams
// are identical. Different increments or offsets may carry a write dependency.
func complexIncrementsCompatibleSIMD(x, y []complex64, incX, incY, ix, iy uintptr) bool {
	if !complexSlicesCompatibleSIMD(x, y) {
		return false
	}
	return len(x) == 0 || len(y) == 0 || unsafe.SliceData(x) != unsafe.SliceData(y) || incX == incY && ix == iy
}
