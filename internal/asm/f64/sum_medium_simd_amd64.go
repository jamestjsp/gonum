// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import "simd/archsimd"

// The caller selects native width >=256. Only complete vectors are loaded.
func sumMediumHardwareSIMD(x []float64) float64 {
	var a, b, c, d archsimd.Float64x4
	for len(x) >= 16 {
		block := x[:16]
		a = archsimd.LoadFloat64x4Array((*[4]float64)(block[:4])).Add(a)
		b = archsimd.LoadFloat64x4Array((*[4]float64)(block[4:8])).Add(b)
		c = archsimd.LoadFloat64x4Array((*[4]float64)(block[8:12])).Add(c)
		d = archsimd.LoadFloat64x4Array((*[4]float64)(block[12:16])).Add(d)
		x = x[16:]
	}
	a = a.Add(b).Add(c.Add(d))
	for len(x) >= 4 {
		a = archsimd.LoadFloat64x4Array((*[4]float64)(x[:4])).Add(a)
		x = x[4:]
	}
	pair := a.GetLo().Add(a.GetHi())
	// No wide values remain live; scalar reduction and tails may use SSE.
	archsimd.ClearAVXUpperBits()
	sum := pair.GetElem(0) + pair.GetElem(1)
	for _, v := range x {
		sum += v
	}
	return sum
}
