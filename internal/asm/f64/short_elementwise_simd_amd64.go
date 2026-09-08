// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

func shortElementwiseHardwareSIMD(n int) bool {
	return n < 64 && !simd.Emulated() && archsimd.X86.AVX2()
}

// Scaling uses four-lane vectors; Linf retains its two-lane native path.
func shortScalHardwareSIMD(n int) bool {
	return shortElementwiseHardwareSIMD(n) && simd.VectorBitSize() >= 256
}

// linfShortHardwareSIMD uses complete two-element loads only. The nonnegative
// checksum detects every NaN difference, independent of the native Max operand
// order, and asks the original recurrence to handle exceptional inputs.
func linfShortHardwareSIMD(x, y []float64) (float64, bool) {
	y = y[:len(x):len(x)]
	n := len(x)
	xp, yp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y))
	mask := archsimd.BroadcastUint64x2(0x7fffffffffffffff)
	var a, b, c, d, checksum archsimd.Float64x2
	for n >= 8 {
		xb, yb := (*[8]float64)(xp), (*[8]float64)(yp)
		v0 := linfDifferenceBlock2SIMD(xb[0:2], yb[0:2], mask)
		v1 := linfDifferenceBlock2SIMD(xb[2:4], yb[2:4], mask)
		v2 := linfDifferenceBlock2SIMD(xb[4:6], yb[4:6], mask)
		v3 := linfDifferenceBlock2SIMD(xb[6:8], yb[6:8], mask)
		a, b, c, d = a.Max(v0), b.Max(v1), c.Max(v2), d.Max(v3)
		checksum = checksum.Add(v0.Add(v1).Add(v2.Add(v3)))
		n -= 8
		if n == 0 {
			break
		}
		xp, yp = unsafe.Add(xp, 64), unsafe.Add(yp, 64)
	}
	a = a.Max(b).Max(c.Max(d))
	for n >= 2 {
		v := linfDifference2SIMD(xp, yp, 0, mask)
		a = a.Max(v)
		checksum = checksum.Add(v)
		n -= 2
		if n == 0 {
			break
		}
		xp, yp = unsafe.Add(xp, 16), unsafe.Add(yp, 16)
	}
	if math.IsNaN(checksum.GetElem(0) + checksum.GetElem(1)) {
		return 0, false
	}
	result := max(a.GetElem(0), a.GetElem(1))
	if n > 0 {
		v := math.Abs(*(*float64)(yp) - *(*float64)(xp))
		if math.IsNaN(v) {
			return 0, false
		}
		result = max(result, v)
	}
	return result, true
}

// The loop guards ensure both elements are within each original slice. No
// pointer is formed for an unconsumed tail or beyond the underlying object.
func linfDifference2SIMD(x, y unsafe.Pointer, offset int, mask archsimd.Uint64x2) archsimd.Float64x2 {
	return archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(y, offset))).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(x, offset)))).ToBits().And(mask).BitsToFloat64()
}

func linfDifferenceBlock2SIMD(x, y []float64, mask archsimd.Uint64x2) archsimd.Float64x2 {
	return archsimd.LoadFloat64x2Array((*[2]float64)(y)).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(x))).ToBits().And(mask).BitsToFloat64()
}

// scalShortHardwareSIMD requires dst and x to be disjoint or exactly in place;
// the caller retains its scalar recurrence for partial overlap.
func scalShortHardwareSIMD(dst []float64, alpha float64, x []float64) {
	dst = dst[:len(x):len(x)]
	n := len(x)
	xp, dp := unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(dst))
	a := archsimd.BroadcastFloat64x4(alpha)
	for n >= 16 {
		xb, db := (*[16]float64)(xp), (*[16]float64)(dp)
		v0 := archsimd.LoadFloat64x4Array((*[4]float64)(xb[0:4])).Mul(a)
		v1 := archsimd.LoadFloat64x4Array((*[4]float64)(xb[4:8])).Mul(a)
		v2 := archsimd.LoadFloat64x4Array((*[4]float64)(xb[8:12])).Mul(a)
		v3 := archsimd.LoadFloat64x4Array((*[4]float64)(xb[12:16])).Mul(a)
		v0.StoreArray((*[4]float64)(db[0:4]))
		v1.StoreArray((*[4]float64)(db[4:8]))
		v2.StoreArray((*[4]float64)(db[8:12]))
		v3.StoreArray((*[4]float64)(db[12:16]))
		n -= 16
		if n == 0 {
			break
		}
		xp, dp = unsafe.Add(xp, 128), unsafe.Add(dp, 128)
	}
	for n >= 4 {
		archsimd.LoadFloat64x4Array((*[4]float64)(xp)).Mul(a).StoreArray((*[4]float64)(dp))
		n -= 4
		if n == 0 {
			break
		}
		xp, dp = unsafe.Add(xp, 32), unsafe.Add(dp, 32)
	}
	if n >= 2 {
		archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Mul(a.GetLo()).StoreArray((*[2]float64)(dp))
		n -= 2
		if n > 0 {
			xp, dp = unsafe.Add(xp, 16), unsafe.Add(dp, 16)
		}
	}
	archsimd.ClearAVXUpperBits()
	if n > 0 {
		*(*float64)(dp) = alpha * *(*float64)(xp)
	}
}

// The native pointer loop also handles long vectors at the 256-bit width.
func scalToHardwareSIMD(n int) bool {
	width := simd.VectorBitSize()
	return (n < 64 || width == 256) && width >= 256 && !simd.Emulated() && archsimd.X86.AVX2()
}
