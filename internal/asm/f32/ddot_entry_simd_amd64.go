// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"simd/archsimd"
)

func DdotUnitarySIMD(x, y []float32) float64 {
	if !canDdotShortSIMD(len(x)) {
		if simd.Emulated() {
			return ddotUnitaryPortableSIMD(x, y)
		}
		return ddotUnitaryHardwareSIMD(x, y)
	}
	y = y[:len(x):len(x)]
	if len(x) < 4 {
		var sum float64
		for i, v := range x {
			sum += float64(v) * float64(y[i])
		}
		return sum
	}
	lastX, lastY := x[len(x)-4:], y[len(x)-4:]
	var acc, acc1 archsimd.Float64x4
	for len(x) >= 8 {
		acc = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4])).ConvertToFloat64()).Add(acc)
		acc1 = archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[4:8])).ConvertToFloat64()).Add(acc1)
		x, y = x[8:], y[8:]
	}
	acc = acc.Add(acc1)
	if len(x) >= 4 {
		acc = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4])).ConvertToFloat64()).Add(acc)
		x, y = x[4:], y[4:]
	}
	if len(x) != 0 {
		xv := maskShortTailSIMD(archsimd.LoadFloat32x4Array((*[4]float32)(lastX)), len(x)).ConvertToFloat64()
		yv := maskShortTailSIMD(archsimd.LoadFloat32x4Array((*[4]float32)(lastY)), len(x)).ConvertToFloat64()
		acc = xv.Mul(yv).Add(acc)
	}
	pair := acc.GetLo().Add(acc.GetHi())
	return pair.GetElem(0) + pair.GetElem(1)
}
