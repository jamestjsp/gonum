// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import "unsafe"

// Keep the existing scalar stride loops outside portable SIMD multiversioning.
func ScalIncSIMD(alpha float64, x []float64, n, incX uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 {
		ScalUnitarySIMD(alpha, x[:n])
		return
	}
	if incX == 0 {
		for ; n > 0; n-- {
			x[0] *= alpha
		}
		return
	}

	if simdPositiveSpan(len(x), n, incX, 0) {
		p := unsafe.Pointer(unsafe.SliceData(x))
		step, step3 := incX*8, incX*24
		for n >= 4 {
			v0, v1 := *(*float64)(p), *(*float64)(unsafe.Add(p, step))
			v2, v3 := *(*float64)(unsafe.Add(p, 2*step)), *(*float64)(unsafe.Add(p, step3))
			*(*float64)(p) = alpha * v0
			*(*float64)(unsafe.Add(p, step)) = alpha * v1
			*(*float64)(unsafe.Add(p, 2*step)) = alpha * v2
			*(*float64)(unsafe.Add(p, step3)) = alpha * v3
			n -= 4
			if n == 0 {
				return
			}
			p = unsafe.Add(p, 4*step)
		}
		for n > 0 {
			*(*float64)(p) *= alpha
			n--
			if n > 0 {
				p = unsafe.Add(p, step)
			}
		}
		return
	}

	var index uintptr
	for ; n >= 4; n -= 4 {
		x[index] *= alpha
		x[index+incX] *= alpha
		x[index+2*incX] *= alpha
		x[index+3*incX] *= alpha
		index += 4 * incX
	}
	for ; n > 0; n-- {
		x[index] *= alpha
		index += incX
	}
}

func ScalIncToSIMD(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr) {
	if n == 0 {
		return
	}
	if incDst == 1 && incX == 1 {
		ScalUnitaryToSIMD(dst[:n], alpha, x[:n])
		return
	}
	if incX == 0 || incDst == 0 || !simdSlicesCompatible(dst, x) || unsafe.SliceData(dst) == unsafe.SliceData(x) {
		var ix, idst uintptr
		for ; n > 0; n-- {
			dst[idst] = alpha * x[ix]
			ix += incX
			idst += incDst
		}
		return
	}

	if simdPositiveSpan(len(x), n, incX, 0) && simdPositiveSpan(len(dst), n, incDst, 0) {
		xp, dp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(dst))
		sx, sd, sx3, sd3 := incX*8, incDst*8, incX*24, incDst*24
		for n >= 4 {
			v0, v1 := *(*float64)(xp), *(*float64)(unsafe.Add(xp, sx))
			v2, v3 := *(*float64)(unsafe.Add(xp, 2*sx)), *(*float64)(unsafe.Add(xp, sx3))
			*(*float64)(dp) = alpha * v0
			*(*float64)(unsafe.Add(dp, sd)) = alpha * v1
			*(*float64)(unsafe.Add(dp, 2*sd)) = alpha * v2
			*(*float64)(unsafe.Add(dp, sd3)) = alpha * v3
			n -= 4
			if n == 0 {
				return
			}
			xp, dp = unsafe.Add(xp, 4*sx), unsafe.Add(dp, 4*sd)
		}
		for n > 0 {
			*(*float64)(dp) = alpha * *(*float64)(xp)
			n--
			if n > 0 {
				xp, dp = unsafe.Add(xp, sx), unsafe.Add(dp, sd)
			}
		}
		return
	}

	var ix, idst uintptr
	for ; n >= 4; n -= 4 {
		dst[idst] = alpha * x[ix]
		dst[idst+incDst] = alpha * x[ix+incX]
		dst[idst+2*incDst] = alpha * x[ix+2*incX]
		dst[idst+3*incDst] = alpha * x[ix+3*incX]
		ix += 4 * incX
		idst += 4 * incDst
	}
	for ; n > 0; n-- {
		dst[idst] = alpha * x[ix]
		ix += incX
		idst += incDst
	}
}
