// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

func GemvTSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	// Validate before scaling y: overlapping inputs retain the original
	// sequential updates, including the effect of scaling on later inputs.
	if n >= 4 && n <= 16 && incX == 1 && incY == 1 && m != 0 && m <= uintptr(len(x)) && n <= uintptr(len(y)) && n <= uintptr(len(a)) && lda >= n && (m <= 1 || m-1 <= (uintptr(len(a))-n)/lda) && !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() && simdMatrixDisjoint(y, x) && simdMatrixDisjoint(y, a) {
		if n == 8 {
			gemvTEightNativeSIMD(m, math.Float64bits(alpha), math.Float64bits(beta), unsafe.Pointer(unsafe.SliceData(a)), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), lda*8)
		} else {
			gemvTSmallNativeSIMD(m, n, math.Float64bits(alpha), math.Float64bits(beta), unsafe.Pointer(unsafe.SliceData(a)), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), lda*8)
		}
		return
	}
	gemvTPortableSIMD(m, n, alpha, a, lda, x, incX, beta, y, incY)
}
