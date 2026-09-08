// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !arm64 && !safe && !noasm && !gccgo

package f64

func gemvTSmallBackendReference(m, n, lda int, alpha float64, a, x []float64, beta float64, y []float64) {
	gemvTSmallReference(m, n, lda, alpha, a, x, beta, y)
}
