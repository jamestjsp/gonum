// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func GemvNSIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	if n == 8 && incX == 1 && incY == 1 {
		done, ok := gemvNShortHardwareSIMD(m, alpha, a, lda, x, beta, y)
		if ok {
			return
		}
		if done > 0 {
			m -= done
			a = a[done*lda:]
			y = y[done:]
		}
	}

	if n != 8 && incX == 1 && incY == 1 {
		done, ok := gemvNBlockHardwareSIMD(m, n, alpha, a, lda, x, beta, y)
		if ok {
			return
		}
		if done > 0 {
			m -= done
			a = a[done*lda:]
			y = y[done:]
		}
	}

	gemvNPortableSIMD(m, n, alpha, a, lda, x, incX, beta, y, incY)
}
