// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo
// +build go1.27,goexperiment.simd,arm64,!safe,!noasm,!gccgo

package f64

import "simd/archsimd"

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

	var sum0, sum1, sum2, sum3 archsimd.Float64x2
	var i int
	for ; i+7 < len(x); i += 8 {
		sum0 = archsimd.LoadFloat64x2(x[i:]).MulAdd(archsimd.LoadFloat64x2(y[i:]), sum0)
		sum1 = archsimd.LoadFloat64x2(x[i+2:]).MulAdd(archsimd.LoadFloat64x2(y[i+2:]), sum1)
		sum2 = archsimd.LoadFloat64x2(x[i+4:]).MulAdd(archsimd.LoadFloat64x2(y[i+4:]), sum2)
		sum3 = archsimd.LoadFloat64x2(x[i+6:]).MulAdd(archsimd.LoadFloat64x2(y[i+6:]), sum3)
	}

	sum0 = sum0.Add(sum1).Add(sum2.Add(sum3))
	for ; i+1 < len(x); i += 2 {
		sum0 = archsimd.LoadFloat64x2(x[i:]).MulAdd(archsimd.LoadFloat64x2(y[i:]), sum0)
	}
	sum = sum0.ConcatAddPairs(sum0).GetElem(0)
	for ; i < len(x); i++ {
		sum += x[i] * y[i]
	}
	return sum
}
