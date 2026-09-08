// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

func l1MultiOldEntryOracle(x []float64) float64 {
	if len(x) >= 128 || simd.Emulated() || !archsimd.X86.AVX2() {
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
