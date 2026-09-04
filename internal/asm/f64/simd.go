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
		simd.LoadFloat64s(dst[i:]).Add(simd.LoadFloat64s(src[i:])).Store(dst[i:])
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
		simd.LoadFloat64s(x[i:]).Add(a).Store(x[i:])
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
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i:]).Mul(a).Add(simd.LoadFloat64s(y[i:])).Store(y[i:])
	}
	for ; i < len(x); i++ {
		y[i] += alpha * x[i]
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
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i:]).Mul(a).Add(simd.LoadFloat64s(y[i:])).Store(dst[i:])
	}
	for ; i < len(x); i++ {
		dst[i] = alpha*x[i] + y[i]
	}
}

func AxpyIncSIMD(alpha float64, x, y []float64, n, incX, incY, ix, iy uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	var xb, yb [32]float64
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = x[ix]
			yb[lane] = y[iy]
			ix += incX
			iy += incY
		}
		simd.LoadFloat64s(xb[:]).Mul(simd.BroadcastFloat64s(alpha)).Add(simd.LoadFloat64s(yb[:])).Store(yb[:])
		write := iy - uintptr(width)*incY
		for lane := 0; lane < width; lane++ {
			y[write] = yb[lane]
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
	width := simd.BroadcastFloat64s(0).Len()
	var xb, yb, out [32]float64
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = x[ix]
			yb[lane] = y[iy]
			ix += incX
			iy += incY
		}
		simd.LoadFloat64s(xb[:]).Mul(simd.BroadcastFloat64s(alpha)).Add(simd.LoadFloat64s(yb[:])).Store(out[:])
		for lane := 0; lane < width; lane++ {
			dst[idst] = out[lane]
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
	var lanes [32]float64
	var sum float64
	var i int
	for ; i+width <= len(src); i += width {
		simd.LoadFloat64s(src[i:]).Store(lanes[:])
		for lane := 1; lane < width; lane++ {
			lanes[lane] += lanes[lane-1]
		}
		simd.LoadFloat64s(lanes[:]).Add(simd.BroadcastFloat64s(sum)).Store(dst[i:])
		sum = dst[i+width-1]
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
	var lanes [32]float64
	product := 1.0
	var i int
	for ; i+width <= len(src); i += width {
		simd.LoadFloat64s(src[i:]).Store(lanes[:])
		for lane := 1; lane < width; lane++ {
			lanes[lane] *= lanes[lane-1]
		}
		simd.LoadFloat64s(lanes[:]).Mul(simd.BroadcastFloat64s(product)).Store(dst[i:])
		product = dst[i+width-1]
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
		simd.LoadFloat64s(dst[i:]).Div(simd.LoadFloat64s(src[i:])).Store(dst[i:])
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
		simd.LoadFloat64s(x[i:]).Div(simd.LoadFloat64s(y[i:])).Store(dst[i:])
	}
	for ; i < len(x); i++ {
		dst[i] = x[i] / y[i]
	}
	return dst
}

func DotUnitarySIMD(x, y []float64) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		acc = simd.LoadFloat64s(x[i:]).Mul(simd.LoadFloat64s(y[i:])).Add(acc)
	}
	sum := reduceF64(acc)
	for ; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}

func DotIncSIMD(x, y []float64, n, incX, incY, ix, iy uintptr) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var xb, yb [32]float64
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			xb[lane] = x[ix]
			yb[lane] = y[iy]
			ix += incX
			iy += incY
		}
		acc = simd.LoadFloat64s(xb[:]).Mul(simd.LoadFloat64s(yb[:])).Add(acc)
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
		acc = simd.LoadFloat64s(x[i:]).Abs().Add(acc)
	}
	sum := reduceF64(acc)
	for ; i < len(x); i++ {
		sum += math.Abs(x[i])
	}
	return sum
}

func L1NormIncSIMD(x []float64, n, incX int) float64 {
	acc := simd.BroadcastFloat64s(0)
	width := acc.Len()
	var values [32]float64
	index := 0
	remaining := n
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			values[lane] = x[index]
			index += incX
		}
		acc = simd.LoadFloat64s(values[:]).Abs().Add(acc)
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
		acc = simd.LoadFloat64s(x[i:]).Sub(simd.LoadFloat64s(y[i:])).Abs().Add(acc)
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
	maximum := math.Abs(y[0] - x[0])
	width := simd.BroadcastFloat64s(0).Len()
	var lanes [32]float64
	i := 1
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(y[i:]).Sub(simd.LoadFloat64s(x[i:])).Abs().Store(lanes[:])
		for _, value := range lanes[:width] {
			if value > maximum || math.IsNaN(maximum) {
				maximum = value
			}
		}
	}
	for ; i < len(x); i++ {
		value := math.Abs(y[i] - x[i])
		if value > maximum || math.IsNaN(maximum) {
			maximum = value
		}
	}
	return maximum
}

func L2NormUnitarySIMD(x []float64) float64 {
	var state f64NormState
	width := simd.BroadcastFloat64s(0).Len()
	var lanes [32]float64
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i:]).Abs().Store(lanes[:])
		for _, value := range lanes[:width] {
			state.add(value)
		}
	}
	for ; i < len(x); i++ {
		state.add(math.Abs(x[i]))
	}
	return state.norm()
}

func L2NormIncSIMD(x []float64, n, incX uintptr) float64 {
	var state f64NormState
	width := simd.BroadcastFloat64s(0).Len()
	var lanes [32]float64
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			lanes[lane] = x[index]
			index += incX
		}
		simd.LoadFloat64s(lanes[:]).Abs().Store(lanes[:])
		for _, value := range lanes[:width] {
			state.add(value)
		}
		remaining -= width
	}
	for ; remaining > 0; remaining-- {
		state.add(math.Abs(x[index]))
		index += incX
	}
	return state.norm()
}

func L2DistanceUnitarySIMD(x, y []float64) float64 {
	var state f64NormState
	width := simd.BroadcastFloat64s(0).Len()
	var lanes [32]float64
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i:]).Sub(simd.LoadFloat64s(y[i:])).Abs().Store(lanes[:])
		for _, value := range lanes[:width] {
			state.add(value)
		}
	}
	for ; i < len(x); i++ {
		state.add(math.Abs(x[i] - y[i]))
	}
	return state.norm()
}

type f64NormState struct {
	scale      float64
	sumSquares float64
}

func (s *f64NormState) add(abs float64) {
	if abs == 0 || math.IsNaN(s.scale) {
		return
	}
	if math.IsNaN(abs) {
		s.scale = math.NaN()
		return
	}
	if s.sumSquares == 0 {
		s.sumSquares = 1
	}
	if s.scale < abs {
		ratio := s.scale / abs
		s.sumSquares = 1 + s.sumSquares*ratio*ratio
		s.scale = abs
		return
	}
	ratio := abs / s.scale
	s.sumSquares += ratio * ratio
}

func (s f64NormState) norm() float64 {
	if math.IsNaN(s.scale) {
		return math.NaN()
	}
	if math.IsInf(s.scale, 1) {
		return math.Inf(1)
	}
	return s.scale * math.Sqrt(s.sumSquares)
}

func ScalUnitarySIMD(alpha float64, x []float64) {
	a := simd.BroadcastFloat64s(alpha)
	width := a.Len()
	var i int
	for ; i+width <= len(x); i += width {
		simd.LoadFloat64s(x[i:]).Mul(a).Store(x[i:])
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
		simd.LoadFloat64s(x[i:]).Mul(a).Store(dst[i:])
	}
	for ; i < len(x); i++ {
		dst[i] = alpha * x[i]
	}
}

func ScalIncSIMD(alpha float64, x []float64, n, incX uintptr) {
	width := simd.BroadcastFloat64s(0).Len()
	a := simd.BroadcastFloat64s(alpha)
	var values [32]float64
	var index uintptr
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			values[lane] = x[index]
			index += incX
		}
		simd.LoadFloat64s(values[:]).Mul(a).Store(values[:])
		write := index - uintptr(width)*incX
		for lane := 0; lane < width; lane++ {
			x[write] = values[lane]
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
	width := simd.BroadcastFloat64s(0).Len()
	a := simd.BroadcastFloat64s(alpha)
	var values [32]float64
	var ix, idst uintptr
	remaining := int(n)
	for remaining >= width {
		for lane := 0; lane < width; lane++ {
			values[lane] = x[ix]
			ix += incX
		}
		simd.LoadFloat64s(values[:]).Mul(a).Store(values[:])
		for lane := 0; lane < width; lane++ {
			dst[idst] = values[lane]
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
	width := acc.Len()
	var i int
	for ; i+width <= len(x); i += width {
		acc = simd.LoadFloat64s(x[i:]).Add(acc)
	}
	sum := reduceF64(acc)
	for ; i < len(x); i++ {
		sum += x[i]
	}
	return sum
}

func reduceF64(value simd.Float64s) float64 {
	var lanes [32]float64
	width := value.Len()
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
	for row := uintptr(0); row < m; row++ {
		AxpyIncSIMD(alpha*x[ix], y, a[row*lda:row*lda+n], n, incY, 1, iy, 0)
		ix += incX
	}
}

func GemvNSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	var ix, iy uintptr
	if int(incX) < 0 {
		ix = uintptr(-int(n-1) * int(incX))
	}
	if int(incY) < 0 {
		iy = uintptr(-int(m-1) * int(incY))
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
