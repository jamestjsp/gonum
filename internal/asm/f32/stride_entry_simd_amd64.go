// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

// Validate complete positive spans once; retain checked fallbacks for aliases
// of logical elements from zero increments and for negative/invalid strides.
func DotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	if n < 4 {
		var sum float32
		for ; n > 0; n-- {
			sum += x[ix] * y[iy]
			ix += incX
			iy += incY
		}
		return sum
	}
	if incX == 1 && incY == 1 {
		return DotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	if simd.Emulated() || !archsimd.X86.AVX() || !positiveStrideSIMD(len(x), ix, n, incX) || !positiveStrideSIMD(len(y), iy, n, incY) {
		return dotIncPortableEntrySIMD(x, y, n, incX, incY, ix, iy)
	}
	xp, yp := unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy])
	sx, sy := incX*4, incY*4
	left := n
	var acc, acc1 archsimd.Float32x4
	for left >= 8 {
		acc = gatherDotStridedPair4(xp, sx).Mul(gatherDotStridedPair4(yp, sy)).Add(acc)
		acc1 = gatherDotStridedPair4(unsafe.Add(xp, 4*sx), sx).Mul(gatherDotStridedPair4(unsafe.Add(yp, 4*sy), sy)).Add(acc1)
		left -= 8
		if left == 0 {
			break
		}
		xp, yp = unsafe.Add(xp, 8*sx), unsafe.Add(yp, 8*sy)
	}
	acc = acc.Add(acc1)
	if left >= 4 {
		acc = gatherDotStridedPair4(xp, sx).Mul(gatherDotStridedPair4(yp, sy)).Add(acc)
		left -= 4
		if left != 0 {
			xp, yp = unsafe.Add(xp, 4*sx), unsafe.Add(yp, 4*sy)
		}
	}
	pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
	sum := pair.GetElem(0) + pair.GetElem(1)
	for left > 0 {
		sum += *(*float32)(xp) * *(*float32)(yp)
		left--
		if left == 0 {
			break
		}
		xp, yp = unsafe.Add(xp, sx), unsafe.Add(yp, sy)
	}
	if math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
		sum = dotIncPortableSIMD(x, y, n, incX, incY, ix, iy)
		if math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
			return dotIncSequentialSIMD(x, y, n, incX, incY, ix, iy)
		}
	}
	return sum
}

// Validate complete positive spans once; retain checked fallbacks for aliases
// of logical elements from zero increments and for negative/invalid strides.
func DdotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	if n < 4 {
		var sum float64
		for ; n > 0; n-- {
			sum += float64(x[ix]) * float64(y[iy])
			ix += incX
			iy += incY
		}
		return sum
	}
	if incX == 1 && incY == 1 {
		return DdotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	if simd.Emulated() || !archsimd.X86.AVX2() || simd.VectorBitSize() < 256 || !positiveStrideSIMD(len(x), ix, n, incX) || !positiveStrideSIMD(len(y), iy, n, incY) {
		return ddotIncPortableEntrySIMD(x, y, n, incX, incY, ix, iy)
	}
	xp, yp := unsafe.Pointer(&x[ix]), unsafe.Pointer(&y[iy])
	sx, sy := incX*4, incY*4
	left := n
	var acc, acc1 archsimd.Float64x4
	for left >= 8 {
		acc = gatherStridedPointer4(xp, sx).ConvertToFloat64().Mul(gatherStridedPointer4(yp, sy).ConvertToFloat64()).Add(acc)
		acc1 = gatherStridedPointer4(unsafe.Add(xp, 4*sx), sx).ConvertToFloat64().Mul(gatherStridedPointer4(unsafe.Add(yp, 4*sy), sy).ConvertToFloat64()).Add(acc1)
		left -= 8
		if left == 0 {
			break
		}
		xp, yp = unsafe.Add(xp, 8*sx), unsafe.Add(yp, 8*sy)
	}
	acc = acc.Add(acc1)
	if left >= 4 {
		acc = gatherStridedPointer4(xp, sx).ConvertToFloat64().Mul(gatherStridedPointer4(yp, sy).ConvertToFloat64()).Add(acc)
		left -= 4
		if left != 0 {
			xp, yp = unsafe.Add(xp, 4*sx), unsafe.Add(yp, 4*sy)
		}
	}
	pair := acc.GetLo().Add(acc.GetHi())
	// The store orders reduction before ClearAVXUpperBits. Pure GetElem
	// operations can otherwise move after it and spill the wide accumulator.
	var lanes [2]float64
	pair.StoreArray(&lanes)
	archsimd.ClearAVXUpperBits()
	sum := lanes[0] + lanes[1]
	for left > 0 {
		sum += float64(*(*float32)(xp)) * float64(*(*float32)(yp))
		left--
		if left == 0 {
			break
		}
		xp, yp = unsafe.Add(xp, sx), unsafe.Add(yp, sy)
	}
	return sum
}

// Pack two independent pairs before combining them. Each scalar load addresses
// a logical element; no increment gaps or bytes beyond the span are read.
func gatherDotStridedPair4(p unsafe.Pointer, inc uintptr) archsimd.Float32x4 {
	var lo, hi archsimd.Uint32x4
	lo = lo.SetElem(0, *(*uint32)(p))
	lo = lo.SetElem(1, *(*uint32)(unsafe.Add(p, inc)))
	hi = hi.SetElem(0, *(*uint32)(unsafe.Add(p, 2*inc)))
	hi = hi.SetElem(1, *(*uint32)(unsafe.Add(p, 3*inc)))
	return lo.AsFloat32x4().ConcatPermuteScalars(0, 1, 4, 5, hi.AsFloat32x4())
}
