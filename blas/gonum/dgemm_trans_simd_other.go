// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !arm64 && !safe && !noasm && !gccgo

package gonum

func dgemmSerialNotTransSIMD(m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, alpha float64) bool {
	return false
}
