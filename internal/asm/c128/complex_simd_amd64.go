// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"simd"
	"simd/archsimd"
)

// The portable API lacks lane permutations. ToArch is resolved while compiling
// each vector-width variant, leaving one shuffle instruction and no type switch.
func complexSwapSIMD(x simd.Float64s) simd.Float64s {
	switch x := x.ToArch().(type) {
	case archsimd.Float64x8:
		return simd.Float64sFromArch(x.ConcatPermuteScalarsGrouped(1, 2, x))
	case archsimd.Float64x4:
		return simd.Float64sFromArch(x.ConcatPermuteScalarsGrouped(1, 2, x))
	case archsimd.Float64x2:
		return simd.Float64sFromArch(x.ConcatPermuteScalars(1, 2, x))
	}
	return complexSwapPortableSIMD(x)
}

// Keep the entire contiguous loop in archsimd: Go 1.27's FromArch conversion
// otherwise leaves vector stack copies inside every portable shuffle operation.
func complexAxpySIMD(dst, x, y []float64, alpha complex128) int {
	switch ar := simd.BroadcastFloat64s(real(alpha)).ToArch().(type) {
	case archsimd.Float64x8:
		return complexAxpy8SIMD(dst, x, y, ar, imag(alpha))
	case archsimd.Float64x4:
		return complexAxpy4SIMD(dst, x, y, ar, imag(alpha))
	case archsimd.Float64x2:
		return complexAxpy2SIMD(dst, x, y, ar, imag(alpha))
	}
	return -1
}

func complexScalSIMD(x []float64, alpha complex128) int {
	switch ar := simd.BroadcastFloat64s(real(alpha)).ToArch().(type) {
	case archsimd.Float64x8:
		return complexScal8SIMD(x, ar, imag(alpha))
	case archsimd.Float64x4:
		return complexScal4SIMD(x, ar, imag(alpha))
	case archsimd.Float64x2:
		return complexScal2SIMD(x, ar, imag(alpha))
	}
	return -1
}

func complexDotSIMD(x, y []float64, conjugate bool) (complex128, int) {
	switch simd.BroadcastFloat64s(0).ToArch().(type) {
	case archsimd.Float64x8:
		return complexDot8SIMD(x, y, conjugate)
	case archsimd.Float64x4:
		return complexDot4SIMD(x, y, conjugate)
	case archsimd.Float64x2:
		return complexDot2SIMD(x, y, conjugate)
	}
	return 0, -1
}

func complexAxpy8SIMD(dst, x, y []float64, ar archsimd.Float64x8, imagAlpha float64) int {
	signs := [8]float64{-imagAlpha, imagAlpha, -imagAlpha, imagAlpha, -imagAlpha, imagAlpha, -imagAlpha, imagAlpha}
	ai := archsimd.LoadFloat64x8(signs[:])
	n := len(x)
	y, dst = y[:n:n], dst[:n:n]
	for len(x) >= 32 {
		x0, x1 := archsimd.LoadFloat64x8(x[:8]), archsimd.LoadFloat64x8(x[8:16])
		x2, x3 := archsimd.LoadFloat64x8(x[16:24]), archsimd.LoadFloat64x8(x[24:32])
		x0.Mul(ar).Add(x0.ConcatPermuteScalarsGrouped(1, 2, x0).Mul(ai)).Add(archsimd.LoadFloat64x8(y[:8])).Store(dst[:8])
		x1.Mul(ar).Add(x1.ConcatPermuteScalarsGrouped(1, 2, x1).Mul(ai)).Add(archsimd.LoadFloat64x8(y[8:16])).Store(dst[8:16])
		x2.Mul(ar).Add(x2.ConcatPermuteScalarsGrouped(1, 2, x2).Mul(ai)).Add(archsimd.LoadFloat64x8(y[16:24])).Store(dst[16:24])
		x3.Mul(ar).Add(x3.ConcatPermuteScalarsGrouped(1, 2, x3).Mul(ai)).Add(archsimd.LoadFloat64x8(y[24:32])).Store(dst[24:32])
		x, y, dst = x[32:], y[32:], dst[32:]
	}
	for len(x) >= 8 {
		xv := archsimd.LoadFloat64x8(x[:8])
		xv.Mul(ar).Add(xv.ConcatPermuteScalarsGrouped(1, 2, xv).Mul(ai)).Add(archsimd.LoadFloat64x8(y[:8])).Store(dst[:8])
		x, y, dst = x[8:], y[8:], dst[8:]
	}
	return n - len(x)
}

func complexScal8SIMD(x []float64, ar archsimd.Float64x8, imagAlpha float64) int {
	signs := [8]float64{-imagAlpha, imagAlpha, -imagAlpha, imagAlpha, -imagAlpha, imagAlpha, -imagAlpha, imagAlpha}
	ai := archsimd.LoadFloat64x8(signs[:])
	n := len(x)
	for len(x) >= 8 {
		xv := archsimd.LoadFloat64x8(x[:8])
		xv.Mul(ar).Add(xv.ConcatPermuteScalarsGrouped(1, 2, xv).Mul(ai)).Store(x[:8])
		x = x[8:]
	}
	return n - len(x)
}

func complexDot8SIMD(x, y []float64, conjugate bool) (complex128, int) {
	n := len(x)
	y = y[:n:n]
	r0, r1 := archsimd.BroadcastFloat64x8(0), archsimd.BroadcastFloat64x8(0)
	r2, r3 := archsimd.BroadcastFloat64x8(0), archsimd.BroadcastFloat64x8(0)
	i0, i1 := archsimd.BroadcastFloat64x8(0), archsimd.BroadcastFloat64x8(0)
	i2, i3 := archsimd.BroadcastFloat64x8(0), archsimd.BroadcastFloat64x8(0)
	for len(x) >= 32 {
		x0, y0 := archsimd.LoadFloat64x8(x[:8]), archsimd.LoadFloat64x8(y[:8])
		x1, y1 := archsimd.LoadFloat64x8(x[8:16]), archsimd.LoadFloat64x8(y[8:16])
		x2, y2 := archsimd.LoadFloat64x8(x[16:24]), archsimd.LoadFloat64x8(y[16:24])
		x3, y3 := archsimd.LoadFloat64x8(x[24:32]), archsimd.LoadFloat64x8(y[24:32])
		r0 = r0.Add(x0.Mul(y0))
		i0 = i0.Add(x0.Mul(y0.ConcatPermuteScalarsGrouped(1, 2, y0)))
		r1 = r1.Add(x1.Mul(y1))
		i1 = i1.Add(x1.Mul(y1.ConcatPermuteScalarsGrouped(1, 2, y1)))
		r2 = r2.Add(x2.Mul(y2))
		i2 = i2.Add(x2.Mul(y2.ConcatPermuteScalarsGrouped(1, 2, y2)))
		r3 = r3.Add(x3.Mul(y3))
		i3 = i3.Add(x3.Mul(y3.ConcatPermuteScalarsGrouped(1, 2, y3)))
		x, y = x[32:], y[32:]
	}
	r0, i0 = r0.Add(r1).Add(r2).Add(r3), i0.Add(i1).Add(i2).Add(i3)
	for len(x) >= 8 {
		xv, yv := archsimd.LoadFloat64x8(x[:8]), archsimd.LoadFloat64x8(y[:8])
		r0 = r0.Add(xv.Mul(yv))
		i0 = i0.Add(xv.Mul(yv.ConcatPermuteScalarsGrouped(1, 2, yv)))
		x, y = x[8:], y[8:]
	}
	var r, im [8]float64
	r0.Store(r[:])
	i0.Store(im[:])
	var rs, is float64
	for i := 0; i < 8; i += 2 {
		if conjugate {
			rs += r[i] + r[i+1]
			is += im[i] - im[i+1]
		} else {
			rs += r[i] - r[i+1]
			is += im[i] + im[i+1]
		}
	}
	return complex(rs, is), n - len(x)
}

func complexAxpy4SIMD(dst, x, y []float64, ar archsimd.Float64x4, imagAlpha float64) int {
	signs := [4]float64{-imagAlpha, imagAlpha, -imagAlpha, imagAlpha}
	ai := archsimd.LoadFloat64x4(signs[:])
	n := len(x)
	y, dst = y[:n:n], dst[:n:n]
	for len(x) >= 16 {
		x0, x1 := archsimd.LoadFloat64x4(x[:4]), archsimd.LoadFloat64x4(x[4:8])
		x2, x3 := archsimd.LoadFloat64x4(x[8:12]), archsimd.LoadFloat64x4(x[12:16])
		x0.Mul(ar).Add(x0.ConcatPermuteScalarsGrouped(1, 2, x0).Mul(ai)).Add(archsimd.LoadFloat64x4(y[:4])).Store(dst[:4])
		x1.Mul(ar).Add(x1.ConcatPermuteScalarsGrouped(1, 2, x1).Mul(ai)).Add(archsimd.LoadFloat64x4(y[4:8])).Store(dst[4:8])
		x2.Mul(ar).Add(x2.ConcatPermuteScalarsGrouped(1, 2, x2).Mul(ai)).Add(archsimd.LoadFloat64x4(y[8:12])).Store(dst[8:12])
		x3.Mul(ar).Add(x3.ConcatPermuteScalarsGrouped(1, 2, x3).Mul(ai)).Add(archsimd.LoadFloat64x4(y[12:16])).Store(dst[12:16])
		x, y, dst = x[16:], y[16:], dst[16:]
	}
	for len(x) >= 4 {
		xv := archsimd.LoadFloat64x4(x[:4])
		xv.Mul(ar).Add(xv.ConcatPermuteScalarsGrouped(1, 2, xv).Mul(ai)).Add(archsimd.LoadFloat64x4(y[:4])).Store(dst[:4])
		x, y, dst = x[4:], y[4:], dst[4:]
	}
	return n - len(x)
}

func complexScal4SIMD(x []float64, ar archsimd.Float64x4, imagAlpha float64) int {
	signs := [4]float64{-imagAlpha, imagAlpha, -imagAlpha, imagAlpha}
	ai := archsimd.LoadFloat64x4(signs[:])
	n := len(x)
	for len(x) >= 4 {
		xv := archsimd.LoadFloat64x4(x[:4])
		xv.Mul(ar).Add(xv.ConcatPermuteScalarsGrouped(1, 2, xv).Mul(ai)).Store(x[:4])
		x = x[4:]
	}
	return n - len(x)
}

func complexDot4SIMD(x, y []float64, conjugate bool) (complex128, int) {
	n := len(x)
	y = y[:n:n]
	r0, r1 := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	r2, r3 := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	i0, i1 := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	i2, i3 := archsimd.BroadcastFloat64x4(0), archsimd.BroadcastFloat64x4(0)
	for len(x) >= 16 {
		x0, y0 := archsimd.LoadFloat64x4(x[:4]), archsimd.LoadFloat64x4(y[:4])
		x1, y1 := archsimd.LoadFloat64x4(x[4:8]), archsimd.LoadFloat64x4(y[4:8])
		x2, y2 := archsimd.LoadFloat64x4(x[8:12]), archsimd.LoadFloat64x4(y[8:12])
		x3, y3 := archsimd.LoadFloat64x4(x[12:16]), archsimd.LoadFloat64x4(y[12:16])
		r0 = r0.Add(x0.Mul(y0))
		i0 = i0.Add(x0.Mul(y0.ConcatPermuteScalarsGrouped(1, 2, y0)))
		r1 = r1.Add(x1.Mul(y1))
		i1 = i1.Add(x1.Mul(y1.ConcatPermuteScalarsGrouped(1, 2, y1)))
		r2 = r2.Add(x2.Mul(y2))
		i2 = i2.Add(x2.Mul(y2.ConcatPermuteScalarsGrouped(1, 2, y2)))
		r3 = r3.Add(x3.Mul(y3))
		i3 = i3.Add(x3.Mul(y3.ConcatPermuteScalarsGrouped(1, 2, y3)))
		x, y = x[16:], y[16:]
	}
	r0, i0 = r0.Add(r1).Add(r2).Add(r3), i0.Add(i1).Add(i2).Add(i3)
	for len(x) >= 4 {
		xv, yv := archsimd.LoadFloat64x4(x[:4]), archsimd.LoadFloat64x4(y[:4])
		r0 = r0.Add(xv.Mul(yv))
		i0 = i0.Add(xv.Mul(yv.ConcatPermuteScalarsGrouped(1, 2, yv)))
		x, y = x[4:], y[4:]
	}
	var r, im [4]float64
	r0.Store(r[:])
	i0.Store(im[:])
	var rs, is float64
	for i := 0; i < 4; i += 2 {
		if conjugate {
			rs += r[i] + r[i+1]
			is += im[i] - im[i+1]
		} else {
			rs += r[i] - r[i+1]
			is += im[i] + im[i+1]
		}
	}
	return complex(rs, is), n - len(x)
}

func complexAxpy2SIMD(dst, x, y []float64, ar archsimd.Float64x2, imagAlpha float64) int {
	signs := [2]float64{-imagAlpha, imagAlpha}
	ai := archsimd.LoadFloat64x2(signs[:])
	n := len(x)
	y, dst = y[:n:n], dst[:n:n]
	for len(x) >= 8 {
		x0, x1 := archsimd.LoadFloat64x2(x[:2]), archsimd.LoadFloat64x2(x[2:4])
		x2, x3 := archsimd.LoadFloat64x2(x[4:6]), archsimd.LoadFloat64x2(x[6:8])
		x0.Mul(ar).Add(x0.ConcatPermuteScalars(1, 2, x0).Mul(ai)).Add(archsimd.LoadFloat64x2(y[:2])).Store(dst[:2])
		x1.Mul(ar).Add(x1.ConcatPermuteScalars(1, 2, x1).Mul(ai)).Add(archsimd.LoadFloat64x2(y[2:4])).Store(dst[2:4])
		x2.Mul(ar).Add(x2.ConcatPermuteScalars(1, 2, x2).Mul(ai)).Add(archsimd.LoadFloat64x2(y[4:6])).Store(dst[4:6])
		x3.Mul(ar).Add(x3.ConcatPermuteScalars(1, 2, x3).Mul(ai)).Add(archsimd.LoadFloat64x2(y[6:8])).Store(dst[6:8])
		x, y, dst = x[8:], y[8:], dst[8:]
	}
	for len(x) >= 2 {
		xv := archsimd.LoadFloat64x2(x[:2])
		xv.Mul(ar).Add(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai)).Add(archsimd.LoadFloat64x2(y[:2])).Store(dst[:2])
		x, y, dst = x[2:], y[2:], dst[2:]
	}
	return n - len(x)
}

func complexScal2SIMD(x []float64, ar archsimd.Float64x2, imagAlpha float64) int {
	signs := [2]float64{-imagAlpha, imagAlpha}
	ai := archsimd.LoadFloat64x2(signs[:])
	n := len(x)
	for len(x) >= 2 {
		xv := archsimd.LoadFloat64x2(x[:2])
		xv.Mul(ar).Add(xv.ConcatPermuteScalars(1, 2, xv).Mul(ai)).Store(x[:2])
		x = x[2:]
	}
	return n - len(x)
}

func complexDot2SIMD(x, y []float64, conjugate bool) (complex128, int) {
	n := len(x)
	y = y[:n:n]
	r0, r1 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	r2, r3 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	i0, i1 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	i2, i3 := archsimd.BroadcastFloat64x2(0), archsimd.BroadcastFloat64x2(0)
	for len(x) >= 8 {
		x0, y0 := archsimd.LoadFloat64x2(x[:2]), archsimd.LoadFloat64x2(y[:2])
		x1, y1 := archsimd.LoadFloat64x2(x[2:4]), archsimd.LoadFloat64x2(y[2:4])
		x2, y2 := archsimd.LoadFloat64x2(x[4:6]), archsimd.LoadFloat64x2(y[4:6])
		x3, y3 := archsimd.LoadFloat64x2(x[6:8]), archsimd.LoadFloat64x2(y[6:8])
		r0 = r0.Add(x0.Mul(y0))
		i0 = i0.Add(x0.Mul(y0.ConcatPermuteScalars(1, 2, y0)))
		r1 = r1.Add(x1.Mul(y1))
		i1 = i1.Add(x1.Mul(y1.ConcatPermuteScalars(1, 2, y1)))
		r2 = r2.Add(x2.Mul(y2))
		i2 = i2.Add(x2.Mul(y2.ConcatPermuteScalars(1, 2, y2)))
		r3 = r3.Add(x3.Mul(y3))
		i3 = i3.Add(x3.Mul(y3.ConcatPermuteScalars(1, 2, y3)))
		x, y = x[8:], y[8:]
	}
	r0, i0 = r0.Add(r1).Add(r2).Add(r3), i0.Add(i1).Add(i2).Add(i3)
	for len(x) >= 2 {
		xv, yv := archsimd.LoadFloat64x2(x[:2]), archsimd.LoadFloat64x2(y[:2])
		r0 = r0.Add(xv.Mul(yv))
		i0 = i0.Add(xv.Mul(yv.ConcatPermuteScalars(1, 2, yv)))
		x, y = x[2:], y[2:]
	}
	var r, im [2]float64
	r0.Store(r[:])
	i0.Store(im[:])
	var rs, is float64
	for i := 0; i < 2; i += 2 {
		if conjugate {
			rs += r[i] + r[i+1]
			is += im[i] - im[i+1]
		} else {
			rs += r[i] - r[i+1]
			is += im[i] + im[i+1]
		}
	}
	return complex(rs, is), n - len(x)
}
