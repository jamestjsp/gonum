// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo
// +build go1.27,goexperiment.simd,arm64,!safe,!noasm,!gccgo

package f64

import (
	"simd/archsimd"
	"unsafe"
)

// DotUnitary is
//
//	for i, v := range x {
//		sum += y[i] * v
//	}
//	return sum
func DotUnitary(x, y []float64) (sum float64) {
	if len(x) < 16 {
		for i, v := range x {
			sum += y[i] * v
		}
		return sum
	}
	_ = y[len(x)-1]
	return dotUnitaryArchSIMD(&x[0], &y[0], uintptr(len(x)))
}

func dotUnitaryArchSIMD(x, y *float64, n uintptr) (sum float64) {
	var sum0, sum1, sum2, sum3 archsimd.Float64x2
	px, py := unsafe.Pointer(x), unsafe.Pointer(y)
	var i uintptr
	for ; i+7 < n; i += 8 {
		xp := unsafe.Add(px, i*8)
		yp := unsafe.Add(py, i*8)
		sum0 = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).MulAdd(archsimd.LoadFloat64x2Array((*[2]float64)(yp)), sum0)
		sum1 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 16))).MulAdd(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 16))), sum1)
		sum2 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 32))).MulAdd(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 32))), sum2)
		sum3 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 48))).MulAdd(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 48))), sum3)
	}

	sum0 = sum0.Add(sum1).Add(sum2.Add(sum3))
	for ; i+1 < n; i += 2 {
		sum0 = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(px, i*8))).MulAdd(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(py, i*8))), sum0)
	}
	sum = sum0.ConcatAddPairs(sum0).GetElem(0)
	for ; i < n; i++ {
		sum += *(*float64)(unsafe.Add(px, i*8)) * *(*float64)(unsafe.Add(py, i*8))
	}
	return sum
}
