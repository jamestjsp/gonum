// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"simd/archsimd"
)

const hardwareWidenSIMD = true

// loadWidenSIMD uses vector conversion where the portable API currently lacks
// widening. The compiler specializes this type switch for the selected width;
// no interface dispatch or scalar conversion remains in AVX hot loops.
func loadWidenSIMD(x []float32) simd.Float64s {
	switch simd.BroadcastFloat64s(0).ToArch().(type) {
	case archsimd.Float64x8:
		return simd.Float64sFromArch(archsimd.LoadFloat32x8Array((*[8]float32)(x)).ConvertToFloat64())
	case archsimd.Float64x4:
		return simd.Float64sFromArch(archsimd.LoadFloat32x4Array((*[4]float32)(x)).ConvertToFloat64())
	default:
		return loadWidenPortableSIMD(x)
	}
}

// canDdotShortSIMD checks a slice length against the cached native admission
// limit. Go 1.27 initializes its Float64x4 accumulators with VPXOR YMM, which
// requires AVX2 even though widening uses AVX.
func canDdotShortSIMD(n int) bool {
	return uint(n) < uint(nativeDdotShortLimitSIMD) && n&7 != 0
}

// DdotUnitarySIMD has already rejected the short native path.
func ddotUnitaryHardwareSIMD(x, y []float32) float64 {
	if !simd.Emulated() {
		switch simd.VectorBitSize() {
		case 512:
			return ddotUnitary512(x, y)
		case 256:
			// A requested width can be capped from native512 even when AVX2
			// is independently disabled. Avoid native and portable256 here.
			if !archsimd.X86.AVX2() {
				return ddotUnitarySequentialSIMD(x, y)
			}
			return ddotUnitary256(x, y)
		}
	}
	return ddotUnitaryPortableSIMD(x, y)
}

func ddotUnitarySequentialSIMD(x, y []float32) float64 {
	y = y[:len(x):len(x)]
	var sum float64
	for i, v := range x {
		sum += float64(v) * float64(y[i])
	}
	return sum
}

// Keep the widening loop entirely in archsimd: Go 1.27's FromArch bridge
// currently spills each converted vector twice when used inside portable loops.
func ddotUnitary256(x, y []float32) float64 {
	var acc, acc1, acc2, acc3 archsimd.Float64x4
	y = y[:len(x):len(x)]
	for len(x) >= 16 {
		acc = archsimd.LoadFloat32x4Array((*[4]float32)(x[0:4])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[0:4])).ConvertToFloat64()).Add(acc)
		acc1 = archsimd.LoadFloat32x4Array((*[4]float32)(x[4:8])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[4:8])).ConvertToFloat64()).Add(acc1)
		acc2 = archsimd.LoadFloat32x4Array((*[4]float32)(x[8:12])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[8:12])).ConvertToFloat64()).Add(acc2)
		acc3 = archsimd.LoadFloat32x4Array((*[4]float32)(x[12:16])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[12:16])).ConvertToFloat64()).Add(acc3)
		x, y = x[16:], y[16:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= 4 {
		acc = archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4])).ConvertToFloat64()).Add(acc)
		x, y = x[4:], y[4:]
	}
	pair := acc.GetLo().Add(acc.GetHi())
	sum := pair.GetElem(0) + pair.GetElem(1)
	for i, v := range x {
		sum += float64(v) * float64(y[i])
	}
	return sum
}

// Keep the widening loop entirely in archsimd: Go 1.27's FromArch bridge
// currently spills each converted vector twice when used inside portable loops.
func ddotUnitary512(x, y []float32) float64 {
	var acc, acc1, acc2, acc3 archsimd.Float64x8
	y = y[:len(x):len(x)]
	for len(x) >= 32 {
		acc = archsimd.LoadFloat32x8Array((*[8]float32)(x[0:8])).ConvertToFloat64().Mul(archsimd.LoadFloat32x8Array((*[8]float32)(y[0:8])).ConvertToFloat64()).Add(acc)
		acc1 = archsimd.LoadFloat32x8Array((*[8]float32)(x[8:16])).ConvertToFloat64().Mul(archsimd.LoadFloat32x8Array((*[8]float32)(y[8:16])).ConvertToFloat64()).Add(acc1)
		acc2 = archsimd.LoadFloat32x8Array((*[8]float32)(x[16:24])).ConvertToFloat64().Mul(archsimd.LoadFloat32x8Array((*[8]float32)(y[16:24])).ConvertToFloat64()).Add(acc2)
		acc3 = archsimd.LoadFloat32x8Array((*[8]float32)(x[24:32])).ConvertToFloat64().Mul(archsimd.LoadFloat32x8Array((*[8]float32)(y[24:32])).ConvertToFloat64()).Add(acc3)
		x, y = x[32:], y[32:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= 8 {
		acc = archsimd.LoadFloat32x8Array((*[8]float32)(x[:8])).ConvertToFloat64().Mul(archsimd.LoadFloat32x8Array((*[8]float32)(y[:8])).ConvertToFloat64()).Add(acc)
		x, y = x[8:], y[8:]
	}
	if len(x) >= 4 {
		product := archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).ConvertToFloat64().Mul(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4])).ConvertToFloat64())
		acc = acc.SetLo(acc.GetLo().Add(product))
		x, y = x[4:], y[4:]
	}
	half := acc.GetLo().Add(acc.GetHi())
	pair := half.GetLo().Add(half.GetHi())
	sum := pair.GetElem(0) + pair.GetElem(1)
	for i, v := range x {
		sum += float64(v) * float64(y[i])
	}
	return sum
}
