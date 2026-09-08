// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

func complexDotShortSIMD(x, y []complex128, conjugate bool) complex128 {
	panic("unreachable native short dot")
}

func complexDscalShortSIMD(alpha float64, x []complex128) { panic("unreachable") }

func complexAxpyShortSIMD(dst, x, y []complex128, alpha complex128) { panic("unreachable") }
func complexScalShortSIMD(alpha complex128, x []complex128)         { panic("unreachable") }
