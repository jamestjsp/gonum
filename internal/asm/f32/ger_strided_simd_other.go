// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

func gerPositiveHardwareSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) bool {
	return false
}

func gerEightStridedHardwareSIMD(m uintptr, alpha float32, x []float32, incX, ix uintptr, y []float32, incY, iy uintptr, a []float32, lda uintptr) {
}
