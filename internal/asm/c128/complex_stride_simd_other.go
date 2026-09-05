// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

func complexNativeSIMD() bool { return false }
func complexAxpyIncNativeSIMD(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr) {
	panic("unreachable")
}
func complexDotIncNativeSIMD(x, y []complex128, n, incX, incY, ix, iy uintptr, conjugate bool) complex128 {
	panic("unreachable")
}

func complexScalIncNativeSIMD(alpha complex128, x []complex128, n, inc uintptr) { panic("unreachable") }
func complexDscalIncNativeSIMD(alpha float64, x []complex128, n, inc uintptr)   { panic("unreachable") }
