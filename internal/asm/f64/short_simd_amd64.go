// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"simd/archsimd"
	"unsafe"
)

func dotShortHardwareSIMD(x, y []float64) float64 {
	y = y[:len(x):len(x)]
	n := len(x)
	xp := unsafe.Pointer(unsafe.SliceData(x))
	yp := unsafe.Pointer(unsafe.SliceData(y))
	var a, b, c, d archsimd.Float64x2
	for n >= 8 {
		xb := (*[8]float64)(xp)
		yb := (*[8]float64)(yp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[0:2]))).Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[2:4]))).Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[4:6]))).Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[6:8]))).Add(d)
		n -= 8
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 64)
		yp = unsafe.Add(yp, 64)
	}
	if n >= 2 {
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yp))).Add(a)
	}
	if n >= 4 {
		b = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 16))).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 16)))).Add(b)
	}
	if n >= 6 {
		c = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 32))).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 32)))).Add(c)
	}
	a = a.Add(b).Add(c.Add(d))
	result := a.GetElem(0) + a.GetElem(1)
	if n&1 != 0 {
		result += *(*float64)(unsafe.Add(xp, (n-1)*8)) * *(*float64)(unsafe.Add(yp, (n-1)*8))
	}
	return result
}
func sumShortHardwareSIMD(x []float64) float64 {
	n := len(x)
	xp := unsafe.Pointer(unsafe.SliceData(x))
	var a, b, c, d archsimd.Float64x2
	for n >= 8 {
		xb := (*[8]float64)(xp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Add(d)
		n -= 8
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 64)
	}
	if n >= 2 {
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Add(a)
	}
	if n >= 4 {
		b = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 16))).Add(b)
	}
	if n >= 6 {
		c = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 32))).Add(c)
	}
	a = a.Add(b).Add(c.Add(d))
	result := a.GetElem(0) + a.GetElem(1)
	if n&1 != 0 {
		result += *(*float64)(unsafe.Add(xp, (n-1)*8))
	}
	return result
}
