// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

func complexNativeSIMD() bool { return false }
func complexAxpyIncNativeSIMD(dst []complex64, incDst, idst uintptr, alpha complex64, x, y []complex64, n, incX, incY, ix, iy uintptr) {
	panic("unreachable")
}
func complexDotIncNativeSIMD(x, y []complex64, n, incX, incY, ix, iy uintptr, conjugate bool) complex64 {
	panic("unreachable")
}

func complexDotShortNativeSIMD(x, y []complex64, conjugate bool) complex64 { panic("unreachable") }
func complexAxpyTailNativeSIMD(dst []complex64, alpha complex64, x, y []complex64) int {
	panic("unreachable")
}
