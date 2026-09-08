// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

func DscalIncSIMD(alpha float64, x []complex128, n, inc uintptr) {
	if n == 0 {
		return
	}
	if inc == 1 {
		DscalUnitarySIMD(alpha, x[:n])
		return
	}
	if inc == 0 {
		for ; n > 0; n-- {
			v := x[0]
			x[0] = complex(alpha*real(v), alpha*imag(v))
		}
		return
	}
	if complexNativeSIMD() {
		complexDscalIncNativeSIMD(alpha, x, n, inc)
		return
	}
	portableDscaleSIMD(alpha, x, n, inc)
}

func ScalIncSIMD(alpha complex128, x []complex128, n, inc uintptr) {
	if n == 0 {
		return
	}
	if inc == 1 {
		ScalUnitarySIMD(alpha, x[:n])
		return
	}
	if inc == 0 {
		for ; n > 0; n-- {
			x[0] *= alpha
		}
		return
	}
	if complexNativeSIMD() {
		complexScalIncNativeSIMD(alpha, x, n, inc)
		return
	}
	portableScaleSIMD(alpha, x, n, inc)
}
