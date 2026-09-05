// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math"
	"simd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha complex128, x, y []complex128) {
	AxpyUnitaryToSIMD(y, alpha, x, y)
}

func AxpyUnitaryToSIMD(dst []complex128, alpha complex128, x, y []complex128) {
	n := len(x)
	if n < 4 || !complexSlicesCompatibleSIMD(x, dst) || !complexSlicesCompatibleSIMD(y, dst) {
		for i, value := range x {
			dst[i] = alpha*value + y[i]
		}
		return
	}
	xf, yf, df := complexFloatsSIMD(x), complexFloatsSIMD(y[:n]), complexFloatsSIMD(dst[:n])
	if i := complexAxpySIMD(df, xf, yf, alpha); i >= 0 {
		for i /= 2; i < n; i++ {
			dst[i] = alpha*x[i] + y[i]
		}
		return
	}
	width := simd.BroadcastFloat64s(0).Len()
	ar := simd.BroadcastFloat64s(real(alpha))
	// Negating even imaginary products implements the real subtraction without FMA.
	ai := simd.BroadcastFloat64s(imag(alpha)).ToBits().Xor(complexEvenSignSIMD()).BitsToFloat64()
	i := 0
	for ; i+width <= len(xf); i += width {
		xv, yv := simd.LoadFloat64s(xf[i:i+width]), simd.LoadFloat64s(yf[i:i+width])
		xv.Mul(ar).Add(complexSwapSIMD(xv).Mul(ai)).Add(yv).Store(df[i : i+width])
	}
	for i /= 2; i < n; i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyIncSIMD(alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	AxpyIncToSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy)
}

func AxpyIncToSIMD(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
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

func DotcUnitarySIMD(x, y []complex128) complex128 {
	return portableDotUnitarySIMD(x, y, true)
}

func DotuUnitarySIMD(x, y []complex128) complex128 {
	return portableDotUnitarySIMD(x, y, false)
}

func portableDotUnitarySIMD(x, y []complex128, conjugate bool) complex128 {
	sum := interleavedDotUnitarySIMD(x, y, conjugate)
	if !math.IsNaN(real(sum)) && !math.IsNaN(imag(sum)) && !math.IsInf(real(sum), 0) && !math.IsInf(imag(sum), 0) {
		return sum
	}
	// Separate component sums can overflow before per-element cancellation.
	sum = 0
	for i, v := range x {
		if conjugate {
			v = conj128(v)
		}
		sum += v * y[i]
	}
	return sum
}

func interleavedDotUnitarySIMD(x, y []complex128, conjugate bool) complex128 {
	if len(x) < 4 {
		var sum complex128
		for i, value := range x {
			if conjugate {
				value = conj128(value)
			}
			sum += value * y[i]
		}
		return sum
	}
	xf, yf := complexFloatsSIMD(x), complexFloatsSIMD(y[:len(x)])
	if sum, i := complexDotSIMD(xf, yf, conjugate); i >= 0 {
		for i /= 2; i < len(x); i++ {
			value := x[i]
			if conjugate {
				value = conj128(value)
			}
			sum += value * y[i]
		}
		return sum
	}
	return portableDotIncSIMD(x, y, uintptr(len(x)), 1, 1, 0, 0, conjugate)
}

func DotcIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
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

func DotuIncSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr) complex128 {
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
func complexFloatsSIMD(x []complex128) []float64 {
	return unsafe.Slice((*float64)(unsafe.Pointer(unsafe.SliceData(x))), 2*len(x))
}

func complexSlicesCompatibleSIMD(x, y []complex128) bool {
	if len(x) == 0 || len(y) == 0 {
		return true
	}
	xp, yp := uintptr(unsafe.Pointer(unsafe.SliceData(x))), uintptr(unsafe.Pointer(unsafe.SliceData(y)))
	const size = unsafe.Sizeof(complex128(0))
	return xp == yp || xp+uintptr(len(x))*size <= yp || yp+uintptr(len(y))*size <= xp
}

func conj128(value complex128) complex128 {
	return complex(real(value), -imag(value))
}

func DscalUnitarySIMD(alpha float64, x []complex128) {
	xf := complexFloatsSIMD(x)
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	i := 0
	for ; i+width <= len(xf); i += width {
		simd.LoadFloat64s(xf[i : i+width]).Mul(a).Store(xf[i : i+width])
	}
	for ; i < len(xf); i++ {
		xf[i] *= alpha
	}
}

func DscalIncSIMD(alpha float64, x []complex128, n, inc uintptr) {
	if n == 0 {
		return
	}
	if inc == 1 {
		DscalUnitarySIMD(alpha, x[:n])
		return
	}
	if inc == 0 {
		for ; n > 0; n-- {
			v := x[0]
			x[0] = complex(alpha*real(v), alpha*imag(v))
		}
		return
	}
	if complexNativeSIMD() {
		complexDscalIncNativeSIMD(alpha, x, n, inc)
		return
	}
	portableDscaleSIMD(alpha, x, n, inc)
}

func ScalUnitarySIMD(alpha complex128, x []complex128) {
	if len(x) < 4 {
		for i := range x {
			x[i] *= alpha
		}
		return
	}
	xf := complexFloatsSIMD(x)
	if i := complexScalSIMD(xf, alpha); i >= 0 {
		for i /= 2; i < len(x); i++ {
			x[i] *= alpha
		}
		return
	}
	portableScaleSIMD(alpha, x, uintptr(len(x)), 1)
}

func ScalIncSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	if n == 0 {
		return
	}
	if inc == 1 {
		ScalUnitarySIMD(alpha, x[:n])
		return
	}
	if inc == 0 {
		for ; n > 0; n-- {
			x[0] *= alpha
		}
		return
	}
	if complexNativeSIMD() {
		complexScalIncNativeSIMD(alpha, x, n, inc)
		return
	}
	portableScaleSIMD(alpha, x, n, inc)
}

func complexEvenSignSIMD() simd.Uint64s {
	bits := make([]uint64, simd.BroadcastUint64s(0).Len())
	for i := 0; i < len(bits); i += 2 {
		bits[i] = 1 << 63
	}
	return simd.LoadUint64s(bits)
}

// Preserve component bits in the emulated and non-AMD64 permutation fallback.
// Keep this slow fallback out of the AMD64 shuffle helper's inlining budget.
//
//go:noinline
func complexSwapPortableSIMD(x simd.Float64s) simd.Float64s {
	bits := make([]uint64, x.Len())
	x.ToBits().Store(bits)
	for i := 0; i < len(bits); i += 2 {
		bits[i], bits[i+1] = bits[i+1], bits[i]
	}
	return simd.LoadUint64s(bits).BitsToFloat64()
}

func complex128MulAdd(ar, ai simd.Float64s, xr, xi, yr, yi, rr, ri []uint64) {
	xrv, xiv := simd.LoadUint64s(xr).BitsToFloat64(), simd.LoadUint64s(xi).BitsToFloat64()
	xrv.Mul(ar).Sub(xiv.Mul(ai)).Add(simd.LoadUint64s(yr).BitsToFloat64()).ToBits().Store(rr)
	xrv.Mul(ai).Add(xiv.Mul(ar)).Add(simd.LoadUint64s(yi).BitsToFloat64()).ToBits().Store(ri)
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

func complex128DotBlock(xr, xi, yr, yi []uint64, realAcc, imagAcc simd.Float64s, conjugate bool) (simd.Float64s, simd.Float64s) {
	xrv, xiv := simd.LoadUint64s(xr).BitsToFloat64(), simd.LoadUint64s(xi).BitsToFloat64()
	yrv, yiv := simd.LoadUint64s(yr).BitsToFloat64(), simd.LoadUint64s(yi).BitsToFloat64()
	if conjugate {
		return xrv.Mul(yrv).Add(xiv.Mul(yiv)).Add(realAcc), xrv.Mul(yiv).Sub(xiv.Mul(yrv)).Add(imagAcc)
	}
	return xrv.Mul(yrv).Sub(xiv.Mul(yiv)).Add(realAcc), xrv.Mul(yiv).Add(xiv.Mul(yrv)).Add(imagAcc)
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

// Equal base addresses are safe for vector staging only when the index streams
// are identical. Different increments or offsets may carry a write dependency.
func complexIncrementsCompatibleSIMD(x, y []complex128, incX, incY, ix, iy uintptr) bool {
	if !complexSlicesCompatibleSIMD(x, y) {
		return false
	}
	return len(x) == 0 || len(y) == 0 || unsafe.SliceData(x) != unsafe.SliceData(y) || incX == incY && ix == iy
}
