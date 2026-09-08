// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"simd/archsimd"
	"unsafe"
)

func gerARM64SIMD(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr) bool {
	if m < 4 || n < 2 || lda < n || incX != 1 || incY != 1 || m > uintptr(len(x)) || n > uintptr(len(y)) {
		return false
	}
	aLen, ok := matrixSpan(m, n, lda)
	if !ok || aLen > uintptr(len(a)) || !simdMatrixDisjoint(a[:aLen], x[:m]) || !simdMatrixDisjoint(a[:aLen], y[:n]) {
		return false
	}
	xData := unsafe.Pointer(unsafe.SliceData(x))
	yData := unsafe.Pointer(unsafe.SliceData(y))
	aData := unsafe.Pointer(unsafe.SliceData(a))
	row := uintptr(0)
	for ; row+4 <= m; row += 4 {
		x0 := alpha * *(*float64)(unsafe.Add(xData, row*8))
		x1 := alpha * *(*float64)(unsafe.Add(xData, (row+1)*8))
		x2 := alpha * *(*float64)(unsafe.Add(xData, (row+2)*8))
		x3 := alpha * *(*float64)(unsafe.Add(xData, (row+3)*8))
		s0 := archsimd.BroadcastFloat64x2(x0)
		s1 := archsimd.BroadcastFloat64x2(x1)
		s2 := archsimd.BroadcastFloat64x2(x2)
		s3 := archsimd.BroadcastFloat64x2(x3)
		a0 := unsafe.Add(aData, row*lda*8)
		a1 := unsafe.Add(a0, lda*8)
		a2 := unsafe.Add(a1, lda*8)
		a3 := unsafe.Add(a2, lda*8)
		col := uintptr(0)
		for ; col+2 <= n; col += 2 {
			yv := gerARM64Load(yData, col)
			gerARM64Store(a0, col, s0.MulAdd(yv, gerARM64Load(a0, col)))
			gerARM64Store(a1, col, s1.MulAdd(yv, gerARM64Load(a1, col)))
			gerARM64Store(a2, col, s2.MulAdd(yv, gerARM64Load(a2, col)))
			gerARM64Store(a3, col, s3.MulAdd(yv, gerARM64Load(a3, col)))
		}
		if col < n {
			yv := *(*float64)(unsafe.Add(yData, col*8))
			*(*float64)(unsafe.Add(a0, col*8)) += x0 * yv
			*(*float64)(unsafe.Add(a1, col*8)) += x1 * yv
			*(*float64)(unsafe.Add(a2, col*8)) += x2 * yv
			*(*float64)(unsafe.Add(a3, col*8)) += x3 * yv
		}
	}
	for ; row < m; row++ {
		scale := alpha * *(*float64)(unsafe.Add(xData, row*8))
		start := row * lda
		AxpyUnitary(scale, y[:n], a[start:start+n])
	}
	return true
}

func gerARM64Load(data unsafe.Pointer, i uintptr) archsimd.Float64x2 {
	return archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(data, i*8)))
}

func gerARM64Store(data unsafe.Pointer, i uintptr, v archsimd.Float64x2) {
	v.StoreArray((*[2]float64)(unsafe.Add(data, i*8)))
}
