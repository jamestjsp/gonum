// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"unsafe"
)

func AddSIMD(dst, src []float64) {
	if !simdSlicesCompatible(dst[:len(src)], src) {
		for i, value := range src {
			dst[i] += value
		}
		return
	}
	width := simd.BroadcastFloat64s(0).Len()
	var i int
	for ; i+width <= len(src); i += width {
		simd.LoadFloat64s(dst[i : i+width]).Add(simd.LoadFloat64s(src[i : i+width])).Store(dst[i : i+width])
	}
	for ; i < len(src); i++ {
		dst[i] += src[i]
	}
}

func AddConstSIMD(alpha float64, x []float64) {
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i : i+width]).Add(a).Store(x[i : i+width])
	}
	for ; i < len(x); i++ {
		x[i] += alpha
	}
}

func AxpyUnitarySIMD(alpha float64, x, y []float64) {
	if !simdSlicesCompatible(x, y[:len(x)]) {
		for i, value := range x {
			y[i] += alpha * value
		}
		return
	}
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	y = y[:len(x):len(x)]
	for len(x) >= width {
		simd.LoadFloat64s(x[:width]).Mul(a).Add(simd.LoadFloat64s(y[:width])).Store(y[:width])
		x, y = x[width:], y[width:]
	}
	for i, value := range x {
		y[i] += alpha * value
	}
}

func AxpyUnitaryToSIMD(dst []float64, alpha float64, x, y []float64) {
	if !simdSlicesCompatible(dst[:len(x)], x) || !simdSlicesCompatible(dst[:len(x)], y[:len(x)]) {
		for i, value := range x {
			dst[i] = alpha*value + y[i]
		}
		return
	}
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	y, dst = y[:len(x):len(x)], dst[:len(x):len(x)]
	for len(x) >= width {
		simd.LoadFloat64s(x[:width]).Mul(a).Add(simd.LoadFloat64s(y[:width])).Store(dst[:width])
		x, y, dst = x[width:], y[width:], dst[width:]
	}
	for i, value := range x {
		dst[i] = alpha*value + y[i]
	}
}

func AxpyIncSIMD(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 && incY == 1 {
		AxpyUnitarySIMD(alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incX == 0 || incY == 0 || !simdSlicesCompatible(x, y) || unsafe.SliceData(x) == unsafe.SliceData(y) {
		for ; n > 0; n-- {
			y[iy] += alpha * x[ix]
			ix += incX
			iy += incY
		}
		return
	}

	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	// Integer staging avoids legacy SSE loads between AVX operations on amd64.
	xb := make([]uint64, width)
	yb := make([]uint64, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = *(*uint64)(unsafe.Pointer(&x[ix]))
			yb[lane] = *(*uint64)(unsafe.Pointer(&y[iy]))
			ix += incX
			iy += incY
		}
		simd.LoadUint64s(xb).BitsToFloat64().Mul(a).Add(simd.LoadUint64s(yb).BitsToFloat64()).ToBits().Store(yb)
		write := iy - uintptr(width)*incY
		for lane := 0; lane < width; lane++ {
			*(*uint64)(unsafe.Pointer(&y[write])) = yb[lane]
			write += incY
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func AxpyIncToSIMD(dst []float64, incDst, idst uintptr, alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	if n == 0 {
		return
	}
	if incDst == 1 && incX == 1 && incY == 1 {
		AxpyUnitaryToSIMD(dst[idst:idst+n], alpha, x[ix:ix+n], y[iy:iy+n])
		return
	}
	if incDst == 0 || incX == 0 || incY == 0 || !simdSlicesCompatible(dst, x) || !simdSlicesCompatible(dst, y) || unsafe.SliceData(dst) == unsafe.SliceData(x) || unsafe.SliceData(dst) == unsafe.SliceData(y) {
		for ; n > 0; n-- {
			dst[idst] = alpha*x[ix] + y[iy]
			idst += incDst
			ix += incX
			iy += incY
		}
		return
	}

	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	xb := make([]uint64, width)
	yb := make([]uint64, width)
	out := make([]uint64, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = *(*uint64)(unsafe.Pointer(&x[ix]))
			yb[lane] = *(*uint64)(unsafe.Pointer(&y[iy]))
			ix += incX
			iy += incY
		}
		simd.LoadUint64s(xb).BitsToFloat64().Mul(a).Add(simd.LoadUint64s(yb).BitsToFloat64()).ToBits().Store(out)
		for lane := 0; lane < width; lane++ {
			*(*uint64)(unsafe.Pointer(&dst[idst])) = out[lane]
			idst += incDst
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		dst[idst] = alpha*x[ix] + y[iy]
		ix += incX
		iy += incY
		idst += incDst
	}
}

func CumSumSIMD(dst, src []float64) []float64 {
	if !simdSlicesCompatible(dst[:len(src)], src) {
		var sum float64
		for i, value := range src {
			sum += value
			dst[i] = sum
		}
		return dst
	}
	width := simd.BroadcastFloat64s(0).Len()
	lanes := make([]float64, width)
	var sum float64
	var i int
	for ; i+width <= len(src); i += width {
		simd.LoadFloat64s(src[i : i+width]).Store(lanes[:])
		// Peel the first step so two-lane vectors have no inner loop.
		if width > 1 {
			lanes[1] += lanes[0]
		}
		for lane := 2; lane < width; lane++ {
			lanes[lane] += lanes[lane-1]
		}
		result := simd.LoadFloat64s(lanes).Add(simd.BroadcastFloat64s(sum))
		sum = lanes[width-1] + sum
		result.Store(dst[i : i+width])
	}
	for ; i < len(src); i++ {
		sum += src[i]
		dst[i] = sum
	}
	return dst
}

func CumProdSIMD(dst, src []float64) []float64 {
	if !simdSlicesCompatible(dst[:len(src)], src) {
		product := 1.0
		for i, value := range src {
			product *= value
			dst[i] = product
		}
		return dst
	}
	width := simd.BroadcastFloat64s(0).Len()
	lanes := make([]float64, width)
	product := 1.0
	var i int
	for ; i+width <= len(src); i += width {
		simd.LoadFloat64s(src[i : i+width]).Store(lanes[:])
		if width > 1 {
			lanes[1] *= lanes[0]
		}
		for lane := 2; lane < width; lane++ {
			lanes[lane] *= lanes[lane-1]
		}
		result := simd.LoadFloat64s(lanes).Mul(simd.BroadcastFloat64s(product))
		product = lanes[width-1] * product
		result.Store(dst[i : i+width])
	}
	for ; i < len(src); i++ {
		product *= src[i]
		dst[i] = product
	}
	return dst
}

func DivSIMD(dst, src []float64) {
	if !simdSlicesCompatible(dst[:len(src)], src) {
		for i, value := range src {
			dst[i] /= value
		}
		return
	}
	width := simd.BroadcastFloat64s(0).Len()
	var i int
	for ; i+width <= len(src); i += width {
		simd.LoadFloat64s(dst[i : i+width]).Div(simd.LoadFloat64s(src[i : i+width])).Store(dst[i : i+width])
	}
	for ; i < len(src); i++ {
		dst[i] /= src[i]
	}
}

func DivToSIMD(dst, x, y []float64) []float64 {
	if !simdSlicesCompatible(dst[:len(x)], x) || !simdSlicesCompatible(dst[:len(x)], y[:len(x)]) {
		for i, value := range x {
			dst[i] = value / y[i]
		}
		return dst
	}
	width := simd.BroadcastFloat64s(0).Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i : i+width]).Div(simd.LoadFloat64s(y[i : i+width])).Store(dst[i : i+width])
	}
	for ; i < len(x); i++ {
		dst[i] = x[i] / y[i]
	}
	return dst
}

func DotUnitarySIMD(x, y []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	y = y[:len(x):len(x)]
	for len(x) >= 4*width {
		xblock, yblock := x[:4*width], y[:4*width]
		acc = simd.LoadFloat64s(xblock[:width]).Mul(simd.LoadFloat64s(yblock[:width])).Add(acc)
		acc1 = simd.LoadFloat64s(xblock[width : 2*width]).Mul(simd.LoadFloat64s(yblock[width : 2*width])).Add(acc1)
		acc2 = simd.LoadFloat64s(xblock[2*width : 3*width]).Mul(simd.LoadFloat64s(yblock[2*width : 3*width])).Add(acc2)
		acc3 = simd.LoadFloat64s(xblock[3*width : 4*width]).Mul(simd.LoadFloat64s(yblock[3*width : 4*width])).Add(acc3)
		x, y = x[4*width:], y[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat64s(x[:width]).Mul(simd.LoadFloat64s(y[:width])).Add(acc)
		x, y = x[width:], y[width:]
	}
	sum := reduceF64(acc)
	for i, value := range x {
		sum += value * y[i]
	}
	return sum
}

func DotIncSIMD(x, y []float64, n, incX, incY, ix, iy uintptr) float64 {
	if n == 0 {
		return 0
	}
	if incX == 1 && incY == 1 {
		return DotUnitarySIMD(x[ix:ix+n], y[iy:iy+n])
	}
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	xb := make([]uint64, width)
	yb := make([]uint64, width)
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = *(*uint64)(unsafe.Pointer(&x[ix]))
			yb[lane] = *(*uint64)(unsafe.Pointer(&y[iy]))
			ix += incX
			iy += incY
		}
		acc = simd.LoadUint64s(xb).BitsToFloat64().Mul(simd.LoadUint64s(yb).BitsToFloat64()).Add(acc)
		remaining -= width
	}
	sum := reduceF64(acc)
	for ; remaining > 0; remaining-- {
		sum += x[ix] * y[iy]
		ix += incX
		iy += incY
	}
	return sum
}

func L1NormSIMD(x []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		acc = simd.LoadFloat64s(x[i : i+width]).Abs().Add(acc)
	}
	sum := reduceF64(acc)
	for ; i < len(x); i++ {
		sum += math.Abs(x[i])
	}
	return sum
}

func L1NormIncSIMD(x []float64, n, incX int) float64 {
	if n <= 0 {
		return 0
	}
	if incX == 1 {
		return L1NormSIMD(x[:n])
	}
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	values := make([]uint64, width)
	index := 0
	remaining := n
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			values[lane] = *(*uint64)(unsafe.Pointer(&x[index]))
			index += incX
		}
		acc = simd.LoadUint64s(values).BitsToFloat64().Abs().Add(acc)
		remaining -= width
	}
	sum := reduceF64(acc)
	for ; remaining > 0; remaining-- {
		sum += math.Abs(x[index])
		index += incX
	}
	return sum
}

func L1DistSIMD(x, y []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		acc = simd.LoadFloat64s(x[i : i+width]).Sub(simd.LoadFloat64s(y[i : i+width])).Abs().Add(acc)
	}
	sum := reduceF64(acc)
	for ; i < len(x); i++ {
		sum += math.Abs(x[i] - y[i])
	}
	return sum
}

func LinfDistSIMD(x, y []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	y = y[:len(x):len(x)]
	acc := simd.BroadcastFloat64s(math.NaN())
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		v := simd.LoadFloat64s(y[i : i+width]).Sub(simd.LoadFloat64s(x[i : i+width])).Abs()
		// Like the scalar recurrence, ignore NaNs unless all values are NaN.
		acc = v.IfElse(v.Greater(acc).Or(acc.NotEqual(acc)), acc)
	}
	lanes := make([]float64, width)
	acc.Store(lanes)
	maximum := math.NaN()
	for _, v := range lanes {
		if v > maximum || math.IsNaN(maximum) {
			maximum = v
		}
	}
	for ; i < len(x); i++ {
		v := math.Abs(y[i] - x[i])
		if v > maximum || math.IsNaN(maximum) {
			maximum = v
		}
	}
	return maximum
}

// Ordinary magnitudes admit a direct sum of squares. If that sum overflows,
// is non-finite, or is too small to bound accumulated underflow error, retry
// with the scaled recurrence. The lower bound leaves at least 2^104 between
// the sum and one rounded-away subnormal per element.
func normSumUsable(sum float64, n int) bool {
	return sum >= float64(n)*0x1p-970 && sum < math.Inf(1)
}

func L2NormUnitarySIMD(x []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	acc1, acc2, acc3 := acc, acc, acc
	corr, corr1, corr2, corr3 := acc, acc, acc, acc
	width := acc.Len()
	i := 0
	for ; i+4*width <= len(x); i += 4 * width {
		v0 := simd.LoadFloat64s(x[i : i+width])
		v1 := simd.LoadFloat64s(x[i+width : i+2*width])
		v2 := simd.LoadFloat64s(x[i+2*width : i+3*width])
		v3 := simd.LoadFloat64s(x[i+3*width : i+4*width])
		acc, corr = normSquareSIMD(v0, acc, corr)
		acc1, corr1 = normSquareSIMD(v1, acc1, corr1)
		acc2, corr2 = normSquareSIMD(v2, acc2, corr2)
		acc3, corr3 = normSquareSIMD(v3, acc3, corr3)
	}
	acc, corr = normMergeSIMD(acc, corr, acc1, corr1)
	acc, corr = normMergeSIMD(acc, corr, acc2, corr2)
	acc, corr = normMergeSIMD(acc, corr, acc3, corr3)
	for ; i+width <= len(x); i += width {
		v := simd.LoadFloat64s(x[i : i+width])
		acc, corr = normSquareSIMD(v, acc, corr)
	}
	sum, correction := normReduceSIMD(acc, corr)
	for ; i < len(x); i++ {
		sum, correction = normSquareScalar(x[i], sum, correction)
	}
	sum += correction
	if normSumUsable(sum, len(x)) {
		return math.Sqrt(sum)
	}
	return l2NormUnitaryScalar(x)
}

func L2NormIncSIMD(x []float64, n, incX uintptr) float64 {
	if n == 0 {
		return 0
	}
	if incX == 1 {
		return L2NormUnitarySIMD(x[:n])
	}
	acc := simd.BroadcastFloat64s(0)
	corr := acc
	width := acc.Len()
	lanes := make([]uint64, width)
	index := uintptr(0)
	remaining := int(n)
	for remaining >= width {
		for lane := range lanes {
			lanes[lane] = *(*uint64)(unsafe.Pointer(&x[index]))
			index += incX
		}
		v := simd.LoadUint64s(lanes).BitsToFloat64()
		acc, corr = normSquareSIMD(v, acc, corr)
		remaining -= width
	}
	sum, correction := normReduceSIMD(acc, corr)
	for ; remaining > 0; remaining-- {
		sum, correction = normSquareScalar(x[index], sum, correction)
		index += incX
	}
	sum += correction
	if normSumUsable(sum, int(n)) {
		return math.Sqrt(sum)
	}
	return l2NormIncScalar(x, n, incX)
}

func L2DistanceUnitarySIMD(x, y []float64) float64 {
	y = y[:len(x):len(x)]
	acc := simd.BroadcastFloat64s(0)
	acc1, acc2, acc3 := acc, acc, acc
	corr, corr1, corr2, corr3 := acc, acc, acc, acc
	width := acc.Len()
	i := 0
	for ; i+4*width <= len(x); i += 4 * width {
		v0 := simd.LoadFloat64s(x[i : i+width]).Sub(simd.LoadFloat64s(y[i : i+width]))
		v1 := simd.LoadFloat64s(x[i+width : i+2*width]).Sub(simd.LoadFloat64s(y[i+width : i+2*width]))
		v2 := simd.LoadFloat64s(x[i+2*width : i+3*width]).Sub(simd.LoadFloat64s(y[i+2*width : i+3*width]))
		v3 := simd.LoadFloat64s(x[i+3*width : i+4*width]).Sub(simd.LoadFloat64s(y[i+3*width : i+4*width]))
		acc, corr = normSquareSIMD(v0, acc, corr)
		acc1, corr1 = normSquareSIMD(v1, acc1, corr1)
		acc2, corr2 = normSquareSIMD(v2, acc2, corr2)
		acc3, corr3 = normSquareSIMD(v3, acc3, corr3)
	}
	acc, corr = normMergeSIMD(acc, corr, acc1, corr1)
	acc, corr = normMergeSIMD(acc, corr, acc2, corr2)
	acc, corr = normMergeSIMD(acc, corr, acc3, corr3)
	for ; i+width <= len(x); i += width {
		v := simd.LoadFloat64s(x[i : i+width]).Sub(simd.LoadFloat64s(y[i : i+width]))
		acc, corr = normSquareSIMD(v, acc, corr)
	}
	sum, correction := normReduceSIMD(acc, corr)
	for ; i < len(x); i++ {
		sum, correction = normSquareScalar(x[i]-y[i], sum, correction)
	}
	sum += correction
	if normSumUsable(sum, len(x)) {
		return math.Sqrt(sum)
	}
	return l2DistanceUnitaryScalar(x, y)
}

// The sum-of-squares fast path compensates summation error. On fused MulAdd
// backends it also recovers product error, following TwoProductFMA and Dot2
// from Ogita, Rump and Oishi (2005):
// https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf
func normSquareSIMD(v, sum, correction simd.Float64s) (simd.Float64s, simd.Float64s) {
	product := v.Mul(v)
	productError := v.MulAdd(v, product.Neg())
	next, sumError := normTwoSumSIMD(sum, product)
	return next, correction.Add(sumError.Add(productError))
}

func normTwoSumSIMD(a, b simd.Float64s) (simd.Float64s, simd.Float64s) {
	sum := a.Add(b)
	bVirtual := sum.Sub(a)
	return sum, a.Sub(sum.Sub(bVirtual)).Add(b.Sub(bVirtual))
}

func normMergeSIMD(sum, correction, other, otherCorrection simd.Float64s) (simd.Float64s, simd.Float64s) {
	next, sumError := normTwoSumSIMD(sum, other)
	return next, correction.Add(otherCorrection.Add(sumError))
}

func normReduceSIMD(sumVector, correctionVector simd.Float64s) (sum, correction float64) {
	width := sumVector.Len()
	sums, corrections := make([]float64, width), make([]float64, width)
	sumVector.Store(sums)
	correctionVector.Store(corrections)
	for i, v := range sums {
		var sumError float64
		sum, sumError = normTwoSumScalar(sum, v)
		correction += sumError + corrections[i]
	}
	return sum, correction
}

func normSquareScalar(v, sum, correction float64) (float64, float64) {
	product := float64(v * v)
	productError := math.FMA(v, v, -product)
	next, sumError := normTwoSumScalar(sum, product)
	return next, correction + (sumError + productError)
}

func normTwoSumScalar(a, b float64) (float64, float64) {
	sum := a + b
	bVirtual := sum - a
	return sum, (a - (sum - bVirtual)) + (b - bVirtual)
}

func ScalUnitarySIMD(alpha float64, x []float64) {
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i : i+width]).Mul(a).Store(x[i : i+width])
	}
	for ; i < len(x); i++ {
		x[i] *= alpha
	}
}

func ScalUnitaryToSIMD(dst []float64, alpha float64, x []float64) {
	if !simdSlicesCompatible(dst[:len(x)], x) {
		for i, value := range x {
			dst[i] = alpha * value
		}
		return
	}
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i : i+width]).Mul(a).Store(dst[i : i+width])
	}
	for ; i < len(x); i++ {
		dst[i] = alpha * x[i]
	}
}

func ScalIncSIMD(alpha float64, x []float64, n, incX uintptr) {
	if n == 0 {
		return
	}
	if incX == 1 {
		ScalUnitarySIMD(alpha, x[:n])
		return
	}
	if incX == 0 {
		for ; n > 0; n-- {
			x[0] *= alpha
		}
		return
	}

	width := simd.BroadcastFloat64s(0).Len()
	a := simd.BroadcastFloat64s(alpha)
	values := make([]uint64, width)
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			values[lane] = *(*uint64)(unsafe.Pointer(&x[index]))
			index += incX
		}
		simd.LoadUint64s(values).BitsToFloat64().Mul(a).ToBits().Store(values)
		write := index - uintptr(width)*incX
		for lane := 0; lane < width; lane++ {
			*(*uint64)(unsafe.Pointer(&x[write])) = values[lane]
			write += incX
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		x[index] *= alpha
		index += incX
	}
}

func ScalIncToSIMD(dst []float64, incDst uintptr, alpha float64, x []float64, n, incX uintptr) {
	if n == 0 {
		return
	}
	if incDst == 1 && incX == 1 {
		ScalUnitaryToSIMD(dst[:n], alpha, x[:n])
		return
	}
	if incX == 0 || incDst == 0 || !simdSlicesCompatible(dst, x) || unsafe.SliceData(dst) == unsafe.SliceData(x) {
		var ix, idst uintptr
		for ; n > 0; n-- {
			dst[idst] = alpha * x[ix]
			ix += incX
			idst += incDst
		}
		return
	}

	width := simd.BroadcastFloat64s(0).Len()
	a := simd.BroadcastFloat64s(alpha)
	values := make([]uint64, width)
	var ix, idst uintptr
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			values[lane] = *(*uint64)(unsafe.Pointer(&x[ix]))
			ix += incX
		}
		simd.LoadUint64s(values).BitsToFloat64().Mul(a).ToBits().Store(values)
		for lane := 0; lane < width; lane++ {
			*(*uint64)(unsafe.Pointer(&dst[idst])) = values[lane]
			idst += incDst
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		dst[idst] = alpha * x[ix]
		ix += incX
		idst += incDst
	}
}

func SumSIMD(x []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	for len(x) >= 4*width {
		acc = simd.LoadFloat64s(x[:width]).Add(acc)
		acc1 = simd.LoadFloat64s(x[width : 2*width]).Add(acc1)
		acc2 = simd.LoadFloat64s(x[2*width : 3*width]).Add(acc2)
		acc3 = simd.LoadFloat64s(x[3*width : 4*width]).Add(acc3)
		x = x[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat64s(x[:width]).Add(acc)
		x = x[width:]
	}
	sum := reduceF64(acc)
	for _, value := range x {
		sum += value
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

func GerSIMD(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr) {
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
		width := simd.BroadcastFloat64s(0).Len()
		cols := int(n)
		row := uintptr(0)
		for ; row+4 <= m; row += 4 {
			a0 := a[row*lda : row*lda+n : row*lda+n]
			a1 := a[(row+1)*lda : (row+1)*lda+n : (row+1)*lda+n]
			a2 := a[(row+2)*lda : (row+2)*lda+n : (row+2)*lda+n]
			a3 := a[(row+3)*lda : (row+3)*lda+n : (row+3)*lda+n]
			s0, s1 := alpha*x[ix], alpha*x[ix+incX]
			s2, s3 := alpha*x[ix+2*incX], alpha*x[ix+3*incX]
			x0, x1 := simd.BroadcastFloat64s(s0), simd.BroadcastFloat64s(s1)
			x2, x3 := simd.BroadcastFloat64s(s2), simd.BroadcastFloat64s(s3)
			yv := y[:cols:cols]
			j := 0
			for ; j+width <= cols; j += width {
				v := simd.LoadFloat64s(yv[j : j+width])
				v.Mul(x0).Add(simd.LoadFloat64s(a0[j : j+width])).Store(a0[j : j+width])
				v.Mul(x1).Add(simd.LoadFloat64s(a1[j : j+width])).Store(a1[j : j+width])
				v.Mul(x2).Add(simd.LoadFloat64s(a2[j : j+width])).Store(a2[j : j+width])
				v.Mul(x3).Add(simd.LoadFloat64s(a3[j : j+width])).Store(a3[j : j+width])
			}
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
			xv := simd.BroadcastFloat64s(scale)
			j := 0
			for ; j+width <= cols; j += width {
				simd.LoadFloat64s(yv[j : j+width]).Mul(xv).Add(simd.LoadFloat64s(av[j : j+width])).Store(av[j : j+width])
			}
			for ; j < cols; j++ {
				v := yv[j]
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

func simdMatrixDisjoint(a, b []float64) bool {
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return aStart+uintptr(len(a))*8 <= bStart || bStart+uintptr(len(b))*8 <= aStart
}

func GemvNSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var ix, iy uintptr
	if int(incX) < 0 {
		ix = uintptr(-int(n-1) * int(incX))
	}
	if int(incY) < 0 {
		iy = uintptr(-int(m-1) * int(incY))
	}
	// Row blocking shares x loads and gives each accumulator an independent
	// dependency chain. Overlapping inputs use the sequential fallback below.
	if incX == 1 && simdMatrixDisjoint(y, x) && simdMatrixDisjoint(y, a) {
		width := simd.BroadcastFloat64s(0).Len()
		cols := int(n)
		row := uintptr(0)
		for ; row+4 <= m; row += 4 {
			a0 := a[row*lda : row*lda+n : row*lda+n]
			a1 := a[(row+1)*lda : (row+1)*lda+n : (row+1)*lda+n]
			a2 := a[(row+2)*lda : (row+2)*lda+n : (row+2)*lda+n]
			a3 := a[(row+3)*lda : (row+3)*lda+n : (row+3)*lda+n]
			xv := x[:cols:cols]
			d0 := simd.BroadcastFloat64s(0)
			d1, d2, d3 := d0, d0, d0
			j := 0
			for ; j+width <= cols; j += width {
				v := simd.LoadFloat64s(xv[j : j+width])
				d0 = simd.LoadFloat64s(a0[j : j+width]).Mul(v).Add(d0)
				d1 = simd.LoadFloat64s(a1[j : j+width]).Mul(v).Add(d1)
				d2 = simd.LoadFloat64s(a2[j : j+width]).Mul(v).Add(d2)
				d3 = simd.LoadFloat64s(a3[j : j+width]).Mul(v).Add(d3)
			}
			s0, s1 := reduceF64(d0), reduceF64(d1)
			s2, s3 := reduceF64(d2), reduceF64(d3)
			for ; j < cols; j++ {
				v := xv[j]
				s0 += a0[j] * v
				s1 += a1[j] * v
				s2 += a2[j] * v
				s3 += a3[j] * v
			}
			if beta == 0 {
				y[iy] = alpha * s0
				y[iy+incY] = alpha * s1
				y[iy+2*incY] = alpha * s2
				y[iy+3*incY] = alpha * s3
			} else {
				y[iy] = beta*y[iy] + alpha*s0
				y[iy+incY] = beta*y[iy+incY] + alpha*s1
				y[iy+2*incY] = beta*y[iy+2*incY] + alpha*s2
				y[iy+3*incY] = beta*y[iy+3*incY] + alpha*s3
			}
			iy += 4 * incY
		}
		for ; row < m; row++ {
			dot := DotUnitarySIMD(x[:cols:cols], a[row*lda:row*lda+n])
			if beta == 0 {
				y[iy] = alpha * dot
			} else {
				y[iy] = beta*y[iy] + alpha*dot
			}
			iy += incY
		}
		return
	}
	for row := uintptr(0); row < m; row++ {
		dot := DotIncSIMD(x, a[row*lda:row*lda+n], n, incX, 1, ix, 0)
		if beta == 0 {
			y[iy] = alpha * dot
		} else {
			y[iy] = beta*y[iy] + alpha*dot
		}
		iy += incY
	}
}

func GemvTSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var ix, iy uintptr
	if int(incX) < 0 {
		ix = uintptr(-int(m-1) * int(incX))
	}
	if int(incY) < 0 {
		iy = uintptr(-int(n-1) * int(incY))
	}
	if beta == 0 {
		index := iy
		for col := uintptr(0); col < n; col++ {
			y[index] = 0
			index += incY
		}
	} else if int(incY) < 0 {
		ScalIncSIMD(beta, y, n, uintptr(-int(incY)))
	} else {
		ScalIncSIMD(beta, y, n, incY)
	}
	// Keep each output vector in registers while applying four matrix rows.
	// This reduces y traffic without changing the order of its updates.
	if incY == 1 && simdMatrixDisjoint(y, x) && simdMatrixDisjoint(y, a) {
		width := simd.BroadcastFloat64s(0).Len()
		cols := int(n)
		row := uintptr(0)
		for ; row+4 <= m; row += 4 {
			a0 := a[row*lda : row*lda+n : row*lda+n]
			a1 := a[(row+1)*lda : (row+1)*lda+n : (row+1)*lda+n]
			a2 := a[(row+2)*lda : (row+2)*lda+n : (row+2)*lda+n]
			a3 := a[(row+3)*lda : (row+3)*lda+n : (row+3)*lda+n]
			s0, s1 := alpha*x[ix], alpha*x[ix+incX]
			s2, s3 := alpha*x[ix+2*incX], alpha*x[ix+3*incX]
			x0, x1 := simd.BroadcastFloat64s(s0), simd.BroadcastFloat64s(s1)
			x2, x3 := simd.BroadcastFloat64s(s2), simd.BroadcastFloat64s(s3)
			yv := y[:cols:cols]
			j := 0
			for ; j+width <= cols; j += width {
				v := simd.LoadFloat64s(yv[j : j+width])
				v = simd.LoadFloat64s(a0[j : j+width]).Mul(x0).Add(v)
				v = simd.LoadFloat64s(a1[j : j+width]).Mul(x1).Add(v)
				v = simd.LoadFloat64s(a2[j : j+width]).Mul(x2).Add(v)
				v = simd.LoadFloat64s(a3[j : j+width]).Mul(x3).Add(v)
				v.Store(yv[j : j+width])
			}
			for ; j < cols; j++ {
				yv[j] += s0 * a0[j]
				yv[j] += s1 * a1[j]
				yv[j] += s2 * a2[j]
				yv[j] += s3 * a3[j]
			}
			ix += 4 * incX
		}
		for ; row < m; row++ {
			av := a[row*lda : row*lda+n : row*lda+n]
			yv := y[:cols:cols]
			scale := alpha * x[ix]
			xv := simd.BroadcastFloat64s(scale)
			for len(yv) >= 4*width {
				av4 := av[:4*width]
				a0 := simd.LoadFloat64s(av4[:width])
				a1 := simd.LoadFloat64s(av4[width : 2*width])
				a2 := simd.LoadFloat64s(av4[2*width : 3*width])
				a3 := simd.LoadFloat64s(av4[3*width:])
				y0 := simd.LoadFloat64s(yv[:width])
				y1 := simd.LoadFloat64s(yv[width : 2*width])
				y2 := simd.LoadFloat64s(yv[2*width : 3*width])
				y3 := simd.LoadFloat64s(yv[3*width : 4*width])
				a0.Mul(xv).Add(y0).Store(yv[:width])
				a1.Mul(xv).Add(y1).Store(yv[width : 2*width])
				a2.Mul(xv).Add(y2).Store(yv[2*width : 3*width])
				a3.Mul(xv).Add(y3).Store(yv[3*width : 4*width])
				yv, av = yv[4*width:], av[4*width:]
			}
			for len(yv) >= width {
				simd.LoadFloat64s(av).Mul(xv).Add(simd.LoadFloat64s(yv)).Store(yv)
				yv, av = yv[width:], av[width:]
			}
			for j, v := range av {
				yv[j] += scale * v
			}
			ix += incX
		}
		return
	}
	for row := uintptr(0); row < m; row++ {
		AxpyIncSIMD(alpha*x[ix], a[row*lda:row*lda+n], y, n, 1, incY, 0, iy)
		ix += incX
	}
}

func simdSlicesCompatible(a, b []float64) bool {
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
