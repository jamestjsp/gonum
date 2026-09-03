// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || !go1.27 || !goexperiment.simd || noasm || gccgo || safe

package c64

func scalUnitary(alpha complex64, x []complex64) {
	for i := range x {
		x[i] *= alpha
	}
}

func scalUnitaryTo(dst []complex64, alpha complex64, x []complex64) {
	for i, v := range x {
		dst[i] = alpha * v
	}
}

func sscalUnitary(alpha float32, x []complex64) {
	for i, v := range x {
		x[i] = complex(real(v)*alpha, imag(v)*alpha)
	}
}
