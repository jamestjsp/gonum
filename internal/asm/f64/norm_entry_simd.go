// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func L2NormUnitarySIMD(x []float64) float64 {
	if len(x) < 128 && normShortHardwareAvailableSIMD() {
		return l2NormShortNativeSIMD(x)
	}
	return l2NormUnitaryPortableSIMD(x)
}
func L2DistanceUnitarySIMD(x, y []float64) float64 {
	if len(x) < 128 && normShortHardwareAvailableSIMD() {
		return l2DistanceShortNativeSIMD(x, y)
	}
	return l2DistanceUnitaryPortableSIMD(x, y)
}
