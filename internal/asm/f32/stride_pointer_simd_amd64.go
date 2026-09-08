// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math/bits"
	"simd/archsimd"
	"unsafe"
)

// Validate the complete monotone span before using unchecked scalar addresses.
// The high product rejects overflow; zero, negative and invalid increments keep
// their original checked path. No pointer may address an increment gap.
func positiveStrideSIMD(length int, start, n, inc uintptr) bool {
	if n == 0 || int(inc) <= 0 || start >= uintptr(length) {
		return false
	}
	hi, span := bits.Mul(uint(n-1), uint(inc))
	return hi == 0 && uintptr(span) < uintptr(length)-start
}

func gatherStridedPointer4(p unsafe.Pointer, inc uintptr) archsimd.Float32x4 {
	var value archsimd.Uint32x4
	value = value.SetElem(0, *(*uint32)(p))
	value = value.SetElem(1, *(*uint32)(unsafe.Add(p, inc)))
	value = value.SetElem(2, *(*uint32)(unsafe.Add(p, 2*inc)))
	value = value.SetElem(3, *(*uint32)(unsafe.Add(p, 3*inc)))
	return value.AsFloat32x4()
}

func axpyIncPositiveSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	xp, yp, dp := unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy]), unsafe.Pointer(&dst[idst])
	incX, incY, incDst = incX*4, incY*4, incDst*4
	for n >= 4 {
		x0, x1 := *(*float32)(xp), *(*float32)(unsafe.Add(xp, incX))
		x2, x3 := *(*float32)(unsafe.Add(xp, 2*incX)), *(*float32)(unsafe.Add(xp, 3*incX))
		*(*float32)(dp) = alpha*x0 + *(*float32)(yp)
		*(*float32)(unsafe.Add(dp, incDst)) = alpha*x1 + *(*float32)(unsafe.Add(yp, incY))
		*(*float32)(unsafe.Add(dp, 2*incDst)) = alpha*x2 + *(*float32)(unsafe.Add(yp, 2*incY))
		*(*float32)(unsafe.Add(dp, 3*incDst)) = alpha*x3 + *(*float32)(unsafe.Add(yp, 3*incY))
		n -= 4
		if n == 0 {
			return
		}
		xp, yp, dp = unsafe.Add(xp, 4*incX), unsafe.Add(yp, 4*incY), unsafe.Add(dp, 4*incDst)
	}
	for n > 0 {
		*(*float32)(dp) = alpha**(*float32)(xp) + *(*float32)(yp)
		n--
		if n == 0 {
			return
		}
		xp, yp, dp = unsafe.Add(xp, incX), unsafe.Add(yp, incY), unsafe.Add(dp, incDst)
	}
}

func dotIncPositiveSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	xp := unsafe.Pointer(&x[ix])
	yp := unsafe.Pointer(&y[iy])
	incX *= 4
	incY *= 4
	var acc, acc1 archsimd.Float32x4
	i := uintptr(0)
	for ; i+8 <= n; i += 8 {
		xb, yb := unsafe.Add(xp, i*incX), unsafe.Add(yp, i*incY)
		acc = gatherStridedPointer4(xb, incX).Mul(gatherStridedPointer4(yb, incY)).Add(acc)
		acc1 = gatherStridedPointer4(unsafe.Add(xb, 4*incX), incX).Mul(gatherStridedPointer4(unsafe.Add(yb, 4*incY), incY)).Add(acc1)
	}
	acc = acc.Add(acc1)
	if i+4 <= n {
		acc = gatherStridedPointer4(unsafe.Add(xp, i*incX), incX).Mul(gatherStridedPointer4(unsafe.Add(yp, i*incY), incY)).Add(acc)
		i += 4
	}
	pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
	sum := pair.GetElem(0) + pair.GetElem(1)
	for ; i < n; i++ {
		sum += (*(*float32)(unsafe.Add(xp, i*incX))) * (*(*float32)(unsafe.Add(yp, i*incY)))
	}
	return sum
}

func ddotIncPositiveSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	xp := unsafe.Pointer(&x[ix])
	yp := unsafe.Pointer(&y[iy])
	incX *= 4
	incY *= 4
	var acc, acc1 archsimd.Float64x4
	i := uintptr(0)
	for ; i+8 <= n; i += 8 {
		xb, yb := unsafe.Add(xp, i*incX), unsafe.Add(yp, i*incY)
		acc = gatherStridedPointer4(xb, incX).ConvertToFloat64().Mul(gatherStridedPointer4(yb, incY).ConvertToFloat64()).Add(acc)
		acc1 = gatherStridedPointer4(unsafe.Add(xb, 4*incX), incX).ConvertToFloat64().Mul(gatherStridedPointer4(unsafe.Add(yb, 4*incY), incY).ConvertToFloat64()).Add(acc1)
	}
	acc = acc.Add(acc1)
	if i+4 <= n {
		acc = gatherStridedPointer4(unsafe.Add(xp, i*incX), incX).ConvertToFloat64().Mul(gatherStridedPointer4(unsafe.Add(yp, i*incY), incY).ConvertToFloat64()).Add(acc)
		i += 4
	}
	pair := acc.GetLo().Add(acc.GetHi())
	sum := pair.GetElem(0) + pair.GetElem(1)
	for ; i < n; i++ {
		sum += float64(*(*float32)(unsafe.Add(xp, i*incX))) * float64(*(*float32)(unsafe.Add(yp, i*incY)))
	}
	return sum
}
