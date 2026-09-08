// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import "simd/archsimd"

// Short float32 reductions use complete 128-bit loads: wide partial loads overread in
// Go 1.27.1, and the portable-to-architecture reduction bridge spills vectors.
// The float32 paths require only AVX, as does native portable 128-bit SIMD.
// The callers bound these float32 inputs below 32 elements. Four independent
// accumulators let the 16/8/4 decomposition merge once after all vector chunks.
func dotShortHardwareSIMD(x, y []float32) float32 {
	y = y[:len(x):len(x)]
	if len(x) < 4 {
		var sum float32
		for i, v := range x {
			sum += v * y[i]
		}
		return sum
	}
	lastX, lastY := x[len(x)-4:], y[len(x)-4:]
	var a, b, c, d archsimd.Float32x4
	if len(x) >= 16 {
		a = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4]))).Add(a)
		b = archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[4:8])))
		c = archsimd.LoadFloat32x4Array((*[4]float32)(x[8:12])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[8:12])))
		d = archsimd.LoadFloat32x4Array((*[4]float32)(x[12:16])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[12:16])))
		x, y = x[16:], y[16:]
	}
	if len(x) >= 8 {
		a = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4]))).Add(a)
		b = archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[4:8]))).Add(b)
		x, y = x[8:], y[8:]
	}
	if len(x) >= 4 {
		c = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4]))).Add(c)
		x, y = x[4:], y[4:]
	}
	if len(x) == 3 {
		tail := archsimd.LoadFloat32x4Array((*[4]float32)(lastX)).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(lastY)))
		d = d.Add(tail.AsUint32x4().SetElem(0, 0).AsFloat32x4())
		acc := a.Add(b).Add(c.Add(d))
		pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
		return pair.GetElem(0) + pair.GetElem(1)
	}
	acc := a.Add(b).Add(c.Add(d))
	pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
	sum := pair.GetElem(0) + pair.GetElem(1)
	for i, v := range x {
		sum += v * y[i]
	}
	return sum
}

func sumShortHardwareSIMD(x []float32) float32 {
	if len(x) < 4 {
		var sum float32
		for _, v := range x {
			sum += v
		}
		return sum
	}
	last := x[len(x)-4:]
	var a, b, c, d archsimd.Float32x4
	if len(x) >= 16 {
		a = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Add(a)
		b = archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8]))
		c = archsimd.LoadFloat32x4Array((*[4]float32)(x[8:12]))
		d = archsimd.LoadFloat32x4Array((*[4]float32)(x[12:16]))
		x = x[16:]
	}
	if len(x) >= 8 {
		a = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Add(a)
		b = archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8])).Add(b)
		x = x[8:]
	}
	if len(x) >= 4 {
		c = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Add(c)
		x = x[4:]
	}
	if len(x) == 3 {
		d = d.Add(archsimd.LoadFloat32x4Array((*[4]float32)(last)).AsUint32x4().SetElem(0, 0).AsFloat32x4())
		acc := a.Add(b).Add(c.Add(d))
		pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
		return pair.GetElem(0) + pair.GetElem(1)
	}
	acc := a.Add(b).Add(c.Add(d))
	pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
	sum := pair.GetElem(0) + pair.GetElem(1)
	for _, v := range x {
		sum += v
	}
	return sum
}

// A complete load ending at the input boundary includes the last one to three
// values. Zero the already-processed lanes in registers; no memory is masked.
func maskShortTailSIMD(v archsimd.Float32x4, n int) archsimd.Float32x4 {
	switch n {
	case 3:
		return v.AsUint32x4().SetElem(0, 0).AsFloat32x4()
	case 2:
		return v.AsUint64x2().SetElem(0, 0).AsFloat32x4()
	default:
		return v.AsUint64x2().SetElem(0, 0).AsUint32x4().SetElem(2, 0).AsFloat32x4()
	}
}

// The caller establishes native256 or wider. Widen before multiplying, and
// keep short tails in vectors instead of seven scalar conversions at width512.
func ddotShortHardwareSIMD(x, y []float32) float64 {
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
