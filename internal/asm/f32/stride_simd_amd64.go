// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"simd/archsimd"
	"unsafe"
)

const hardwareStridedSIMD = true

// Gather only the requested scalar elements. Integer inserts keep their bits
// intact and avoid legacy SSE loads between AVX operations. A full vector load
// followed by Masked would also read the unrelated increment gaps.
func gatherStrided4(x []float32, ix, inc uintptr) archsimd.Float32x4 {
	var value archsimd.Uint32x4
	value = value.SetElem(0, *(*uint32)(unsafe.Pointer(&x[ix])))
	value = value.SetElem(1, *(*uint32)(unsafe.Pointer(&x[ix+inc])))
	value = value.SetElem(2, *(*uint32)(unsafe.Pointer(&x[ix+2*inc])))
	value = value.SetElem(3, *(*uint32)(unsafe.Pointer(&x[ix+3*inc])))
	return value.AsFloat32x4()
}

func axpyIncHardwareSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) bool {
	// Native portable 128-bit vectors require only AVX, while this
	// register broadcast requires AVX2 in Go 1.27.1.
	if !archsimd.X86.AVX2() {
		return false
	}
	a := archsimd.BroadcastFloat32x4(alpha)
	for n >= 4 {
		xv := gatherStrided4(x, ix, incX)
		yv := gatherStrided4(y, iy, incY)
		out := xv.Mul(a).Add(yv).AsUint32x4()
		*(*uint32)(unsafe.Pointer(&dst[idst])) = out.GetElem(0)
		*(*uint32)(unsafe.Pointer(&dst[idst+incDst])) = out.GetElem(1)
		*(*uint32)(unsafe.Pointer(&dst[idst+2*incDst])) = out.GetElem(2)
		*(*uint32)(unsafe.Pointer(&dst[idst+3*incDst])) = out.GetElem(3)
		ix += 4 * incX
		iy += 4 * incY
		idst += 4 * incDst
		n -= 4
	}
	for ; n > 0; n-- {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
	return true
}

func dotIncHardwareSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) (float32, bool) {
	var acc, acc1 archsimd.Float32x4
	for n >= 8 {
		acc = gatherStrided4(x, ix, incX).Mul(gatherStrided4(y, iy, incY)).Add(acc)
		acc1 = gatherStrided4(x, ix+4*incX, incX).Mul(gatherStrided4(y, iy+4*incY, incY)).Add(acc1)
		ix += 8 * incX
		iy += 8 * incY
		n -= 8
	}
	acc = acc.Add(acc1)
	if n >= 4 {
		acc = gatherStrided4(x, ix, incX).Mul(gatherStrided4(y, iy, incY)).Add(acc)
		ix += 4 * incX
		iy += 4 * incY
		n -= 4
	}
	pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
	sum := pair.GetElem(0) + pair.GetElem(1)
	for ; n > 0; n-- {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum, true
}

func ddotIncHardwareSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) (float64, bool) {
	if simd.VectorBitSize() < 256 {
		return 0, false
	}
	var acc, acc1 archsimd.Float64x4
	for n >= 8 {
		acc = gatherStrided4(x, ix, incX).ConvertToFloat64().Mul(gatherStrided4(y, iy, incY).ConvertToFloat64()).Add(acc)
		acc1 = gatherStrided4(x, ix+4*incX, incX).ConvertToFloat64().Mul(gatherStrided4(y, iy+4*incY, incY).ConvertToFloat64()).Add(acc1)
		ix += 8 * incX
		iy += 8 * incY
		n -= 8
	}
	acc = acc.Add(acc1)
	if n >= 4 {
		acc = gatherStrided4(x, ix, incX).ConvertToFloat64().Mul(gatherStrided4(y, iy, incY).ConvertToFloat64()).Add(acc)
		ix += 4 * incX
		iy += 4 * incY
		n -= 4
	}
	pair := acc.GetLo().Add(acc.GetHi())
	sum := pair.GetElem(0) + pair.GetElem(1)
	for ; n > 0; n-- {
		sum += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
	}
	return sum, true
}
