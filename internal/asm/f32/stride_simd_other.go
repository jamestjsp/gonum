// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

const hardwareStridedSIMD = false

func axpyIncHardwareSIMD(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr) bool {
	return false
}
func dotIncHardwareSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) (float32, bool) {
	return 0, false
}
func ddotIncHardwareSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) (float64, bool) {
	return 0, false
}
