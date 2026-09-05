// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"simd"
	"unsafe"
)

func AxpyUnitarySIMD(alpha float32, x, y []float32) {
	if !simdSlicesCompatible(x, y[:len(x)]) {
		for i, value := range x {
			y[i] += alpha * value
		}
		return
	}
	a := simd.BroadcastFloat32s(alpha)
	width := a.Len()
	y = y[:len(x):len(x)]
	for len(x) >= width {
		simd.LoadFloat32s(x[:width]).Mul(a).Add(simd.LoadFloat32s(y[:width])).Store(y[:width])
		x, y = x[width:], y[width:]
	}
	if hardwareStridedSIMD && !simd.Emulated() && width >= 16 && len(x) >= 8 {
		for len(x) >= 4 {
			axpyTailSIMD(y, alpha, x, y)
			x, y = x[4:], y[4:]
		}
	}

	// Go 1.27.1 partial SIMD loads can read beyond the slice. Keep tails
	// scalar after any complete narrow-vector chunks.
	for i, value := range x {
		y[i] += alpha * value
	}
}

func AxpyUnitaryToSIMD(dst []float32, alpha float32, x, y []float32) {
	if !simdSlicesCompatible(dst[:len(x)], x) || !simdSlicesCompatible(dst[:len(x)], y[:len(x)]) {
		for i, value := range x {
			dst[i] = alpha*value + y[i]
		}
		return
	}
	a := simd.BroadcastFloat32s(alpha)
	width := a.Len()
	y, dst = y[:len(x):len(x)], dst[:len(x):len(x)]
	for len(x) >= width {
		simd.LoadFloat32s(x[:width]).Mul(a).Add(simd.LoadFloat32s(y[:width])).Store(dst[:width])
		x, y, dst = x[width:], y[width:], dst[width:]
	}
	if hardwareStridedSIMD && !simd.Emulated() && width >= 16 && len(x) >= 8 {
		for len(x) >= 4 {
			axpyTailSIMD(dst, alpha, x, y)
			x, y, dst = x[4:], y[4:], dst[4:]
		}
	}

	// Go 1.27.1 partial SIMD loads can read beyond the slice. Keep tails
	// scalar after any complete narrow-vector chunks.
	for i, value := range x {
		dst[i] = alpha*value + y[i]
	}
}

func AxpyIncSIMD(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		AxpyUnitarySIMD(alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if !hardwareStridedSIMD || simd.Emulated() {
		axpyIncPortableSIMD(alpha, x, y, n, incX, incY, ix, iy)
		return
	}

	if n >= 4 && incY != 0 && simdMatrixDisjoint(x, y) && axpyIncHardwareSIMD(y, incY, iy, alpha, x, y, n, incX, incY, ix, iy) {
		return
	}

	// Direct scalar updates avoid packing and scattering scratch vectors.
	// Keep writes in order for zero increments and overlapping slices.
	for n >= 4 {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
		n -= 4
	}
	for ; n > 0; n-- {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func AxpyIncToSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incDst == 1 && incX == 1 && incY == 1 {
		AxpyUnitaryToSIMD(dst[idst:idst+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if !hardwareStridedSIMD || simd.Emulated() {
		axpyIncToPortableSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy)
		return
	}

	if n >= 4 && incDst != 0 && simdMatrixDisjoint(dst, x) && simdMatrixDisjoint(dst, y) && axpyIncHardwareSIMD(dst, incDst, idst, alpha, x, y, n, incX, incY, ix, iy) {
		return
	}

	// Direct scalar updates avoid packing and scattering scratch vectors.
	// Keep writes in order for zero increments and overlapping slices.
	for n >= 4 {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
		n -= 4
	}
	for ; n > 0; n-- {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

func DotUnitarySIMD(x, y []float32) float32 {
	x0, y0 := x, y
	acc := simd.BroadcastFloat32s(0)
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	y = y[:len(x):len(x)]
	for len(x) >= 4*width {
		xblock, yblock := x[:4*width], y[:4*width]
		acc = simd.LoadFloat32s(xblock[:width]).Mul(simd.LoadFloat32s(yblock[:width])).Add(acc)
		acc1 = simd.LoadFloat32s(xblock[width : 2*width]).Mul(simd.LoadFloat32s(yblock[width : 2*width])).Add(acc1)
		acc2 = simd.LoadFloat32s(xblock[2*width : 3*width]).Mul(simd.LoadFloat32s(yblock[2*width : 3*width])).Add(acc2)
		acc3 = simd.LoadFloat32s(xblock[3*width : 4*width]).Mul(simd.LoadFloat32s(yblock[3*width : 4*width])).Add(acc3)
		x, y = x[4*width:], y[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat32s(x[:width]).Mul(simd.LoadFloat32s(y[:width])).Add(acc)
		x, y = x[width:], y[width:]
	}
	sum := reduceF32(acc)
	for i, value := range x {
		sum += value * y[i]
	}
	if hardwareStridedSIMD && !simd.Emulated() && math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
		sum = dotUnitaryOriginalSIMD(x0, y0)
		if math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
			return dotIncSequentialSIMD(x0, y0, uintptr(len(x0)), 1, 1, 0, 0)
		}
	}
	return sum
}

func DotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return DotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	if !hardwareStridedSIMD || simd.Emulated() {
		return dotIncPortableSIMD(x, y, n, incX, incY, ix, iy)
	}

	if n >= 4 {
		if result, ok := dotIncHardwareSIMD(x, y, n, incX, incY, ix, iy); ok {
			if math.Float32bits(result)&0x7f800000 == 0x7f800000 {
				// The original width may cancel values that overflow both
				// the new grouping and a sequential accumulation.
				result = dotIncPortableSIMD(x, y, n, incX, incY, ix, iy)
				if math.Float32bits(result)&0x7f800000 == 0x7f800000 {
					return dotIncSequentialSIMD(x, y, n, incX, incY, ix, iy)
				}
			}
			return result
		}
	}

	var sum0, sum1, sum2, sum3 float32
	for n >= 4 {
		sum0 += x[ix] * y[iy]
		ix += incX
		iy += incY
		sum1 += x[ix] * y[iy]
		ix += incX
		iy += incY
		sum2 += x[ix] * y[iy]
		ix += incX
		iy += incY
		sum3 += x[ix] * y[iy]
		ix += incX
		iy += incY
		n -= 4
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	for ; n > 0; n-- {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func loadWidenPortableSIMD(x []float32) simd.Float64s {
	width := simd.BroadcastFloat64s(0).Len()
	lanes := make([]float64, width)
	for i, v := range x[:width] {
		lanes[i] = float64(v)
	}
	return simd.LoadFloat64s(lanes)
}

func DdotUnitarySIMD(x, y []float32) float64 {
	if simd.Emulated() {
		return ddotUnitaryPortableSIMD(x, y)
	}
	return ddotUnitaryHardwareSIMD(x, y)
}

func ddotUnitaryPortableSIMD(x, y []float32) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	xb, yb := make([]float64, width), make([]float64, width)
	y = y[:len(x):len(x)]
	for len(x) >= width {
		for lane, value := range x[:width] {
			xb[lane] = float64(value)
			yb[lane] = float64(y[:width][lane])
		}
		acc = simd.LoadFloat64s(xb[:]).Mul(simd.LoadFloat64s(yb[:])).Add(acc)
		x, y = x[width:], y[width:]
	}
	sum := reduceF64(acc)
	for i, value := range x {
		sum += float64(value) * float64(y[i])
	}
	return sum
}

func ddotIncPortableSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	xb, yb := make([]float64, width), make([]float64, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = float64(x[ix])
			yb[lane] = float64(y[iy])
			ix += incX
			iy += incY
		}
		acc = simd.LoadFloat64s(xb[:]).Mul(simd.LoadFloat64s(yb[:])).Add(acc)
		remaining -= width
	}
	sum := reduceF64(acc)
	for ; remaining > 0; remaining-- {
		sum += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
	}
	return sum
}

func DdotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return DdotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	if !hardwareStridedSIMD || simd.Emulated() {
		return ddotIncPortableSIMD(x, y, n, incX, incY, ix, iy)
	}

	if n >= 4 {
		if result, ok := ddotIncHardwareSIMD(x, y, n, incX, incY, ix, iy); ok {
			return result
		}
	}

	var sum0, sum1, sum2, sum3 float64
	for n >= 4 {
		sum0 += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
		sum1 += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
		sum2 += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
		sum3 += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
		n -= 4
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	for ; n > 0; n-- {
		sum += float64(x[ix]) * float64(y[iy])
		ix += incX
		iy += incY
	}
	return sum
}

func SumSIMD(x []float32) float32 {
	x0 := x
	acc := simd.BroadcastFloat32s(0)
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	for len(x) >= 4*width {
		acc = simd.LoadFloat32s(x[:width]).Add(acc)
		acc1 = simd.LoadFloat32s(x[width : 2*width]).Add(acc1)
		acc2 = simd.LoadFloat32s(x[2*width : 3*width]).Add(acc2)
		acc3 = simd.LoadFloat32s(x[3*width : 4*width]).Add(acc3)
		x = x[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat32s(x[:width]).Add(acc)
		x = x[width:]
	}
	sum := reduceF32(acc)
	for _, value := range x {
		sum += value
	}
	if hardwareStridedSIMD && !simd.Emulated() && math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
		sum = sumOriginalSIMD(x0)
		if math.Float32bits(sum)&0x7f800000 == 0x7f800000 {
			return sumSequentialSIMD(x0)
		}
	}
	return sum
}

func reduceF32Portable(value simd.Float32s) float32 {
	width := value.Len()
	lanes := make([]float32, width)
	value.Store(lanes[:])
	var sum float32
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}

func reduceF64(value simd.Float64s) float64 {
	width := value.Len()
	lanes := make([]float64, width)
	value.Store(lanes[:])
	var sum float64
	for _, lane := range lanes[:width] {
		sum += lane
	}
	return sum
}

func GerSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) {
	var ix, iy uintptr
	if int(incX) < 0 {
		ix = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		iy = uintptr(-int(n-1) * int(incY))
	}
	// Reuse each vector of y across four independent rows. Retain the
	// sequential path when a write can change a later input.
	if incY == 1 && simdMatrixDisjoint(a, x) && simdMatrixDisjoint(a, y) {
		width := simd.BroadcastFloat32s(0).Len()
		cols := int(n)
		row := uintptr(0)
		for ; row+4 <= m; row += 4 {
			a0 := a[row*lda : row*lda+n : row*lda+n]
			a1 := a[(row+1)*lda : (row+1)*lda+n : (row+1)*lda+n]
			a2 := a[(row+2)*lda : (row+2)*lda+n : (row+2)*lda+n]
			a3 := a[(row+3)*lda : (row+3)*lda+n : (row+3)*lda+n]
			s0, s1 := alpha*x[ix], alpha*x[ix+incX]
			s2, s3 := alpha*x[ix+2*incX], alpha*x[ix+3*incX]
			x0, x1 := simd.BroadcastFloat32s(s0), simd.BroadcastFloat32s(s1)
			x2, x3 := simd.BroadcastFloat32s(s2), simd.BroadcastFloat32s(s3)
			yv := y[:cols:cols]
			j := 0
			for ; j+width <= cols; j += width {
				v := simd.LoadFloat32s(yv[j : j+width])
				v.Mul(x0).Add(simd.LoadFloat32s(a0[j : j+width])).Store(a0[j : j+width])
				v.Mul(x1).Add(simd.LoadFloat32s(a1[j : j+width])).Store(a1[j : j+width])
				v.Mul(x2).Add(simd.LoadFloat32s(a2[j : j+width])).Store(a2[j : j+width])
				v.Mul(x3).Add(simd.LoadFloat32s(a3[j : j+width])).Store(a3[j : j+width])
			}
			if hardwareStridedSIMD && !simd.Emulated() && width >= 8 && cols-j >= 4 {
				j += gerTailSIMD(a0[j:], a1[j:], a2[j:], a3[j:], yv[j:], s0, s1, s2, s3)
			}
			// Partial SIMD loads can cross the end of the final matrix row
			// in Go 1.27.1. Scalar tails also preserve the non-AMD64 path.
			for ; j < cols; j++ {
				v := yv[j]
				a0[j] += s0 * v
				a1[j] += s1 * v
				a2[j] += s2 * v
				a3[j] += s3 * v
			}
			ix += 4 * incX
		}
		for ; row < m; row++ {
			av := a[row*lda : row*lda+n : row*lda+n]
			yv := y[:cols:cols]
			scale := alpha * x[ix]
			xv := simd.BroadcastFloat32s(scale)
			for len(yv) >= width {
				simd.LoadFloat32s(yv).Mul(xv).Add(simd.LoadFloat32s(av)).Store(av)
				yv, av = yv[width:], av[width:]
			}
			for j, v := range yv {
				av[j] += scale * v
			}
			ix += incX
		}
		return
	}
	for row := uintptr(0); row < m; row++ {
		AxpyIncSIMD(alpha*x[ix], y, a[row*lda:row*lda+n], n, incY, 1, iy, 0)
		ix += incX
	}
}

func simdMatrixDisjoint(a, b []float32) bool {
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aStart+uintptr(len(a))*4 <= bStart || bStart+uintptr(len(b))*4 <= aStart
}

func simdSlicesCompatible(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if aStart == bStart {
		return true
	}
	aEnd := aStart + uintptr(len(a))*unsafe.Sizeof(a[0])
	bEnd := bStart + uintptr(len(b))*unsafe.Sizeof(b[0])
	return aEnd <= bStart || bEnd <= aStart
}

// A different reduction tree can overflow finite values before cancellation.
// Keep the exceptional retry outside the usual vector path.
func dotIncSequentialSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	var sum float32
	for ; n > 0; n-- {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func sumSequentialSIMD(x []float32) float32 {
	var sum float32
	for _, v := range x {
		sum += v
	}
	return sum
}
