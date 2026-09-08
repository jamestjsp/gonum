// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func AxpyUnitarySIMD(alpha float64, x, y []float64) { axpyUnitaryPortableSIMD(alpha, x, y) }
func AxpyUnitaryToSIMD(dst []float64, alpha float64, x, y []float64) {
	axpyUnitaryToPortableSIMD(dst, alpha, x, y)
}
