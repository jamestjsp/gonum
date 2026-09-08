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

func L1NormSIMD(x []float64) float64 {
	if len(x) >= 128 {
		if len(x) > 1<<30 || simd.Emulated() || !archsimd.X86.AVX2() {
			return l1NormPortableSIMD(x)
		}
		return l1NormLongSIMD(x)
	}
	if simd.Emulated() || !archsimd.X86.AVX2() {
		return l1NormPortableSIMD(x)
	}
	n := len(x)
	xp := unsafe.Pointer(unsafe.SliceData(x))
	var a, b, c, d archsimd.Float64x2
	for n >= 8 {
		xb := (*[8]float64)(xp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Abs().Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Abs().Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Abs().Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Abs().Add(d)
		n -= 8
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 64)
	}
	a = a.Add(b).Add(c.Add(d))
	for n >= 2 {
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Abs().Add(a)
		n -= 2
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 16)
	}
	sum := a.GetElem(0) + a.GetElem(1)
	if n > 0 {
		sum += math.Abs(*(*float64)(xp))
	}
	if math.Float64bits(sum)&0x7ff0000000000000 != 0x7ff0000000000000 {
		return sum
	}
	return l1NormPortableSIMD(x)
}
func L1DistSIMD(x, y []float64) float64 {
	if len(x) >= 128 || simd.Emulated() || !archsimd.X86.AVX2() {
		return l1DistPortableSIMD(x, y)
	}
	y = y[:len(x):len(x)]
	n := len(x)
	xp := unsafe.Pointer(unsafe.SliceData(x))
	yp := unsafe.Pointer(unsafe.SliceData(y))
	var a, b, c, d archsimd.Float64x2
	for n >= 8 {
		xb := (*[8]float64)(xp)
		yb := (*[8]float64)(yp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(yb[0:2]))).Abs().Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(yb[2:4]))).Abs().Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(yb[4:6]))).Abs().Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(yb[6:8]))).Abs().Add(d)
		n -= 8
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 64)
		yp = unsafe.Add(yp, 64)
	}
	a = a.Add(b).Add(c.Add(d))
	for n >= 2 {
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Sub(archsimd.LoadFloat64x2Array((*[2]float64)(yp))).Abs().Add(a)
		n -= 2
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 16)
		yp = unsafe.Add(yp, 16)
	}
	sum := a.GetElem(0) + a.GetElem(1)
	if n > 0 {
		sum += math.Abs(*(*float64)(xp) - *(*float64)(yp))
	}
	if math.Float64bits(sum)&0x7ff0000000000000 != 0x7ff0000000000000 {
		return sum
	}
	return l1DistPortableSIMD(x, y)
}
