// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func gemvTEightHardwareSIMD(m uintptr, alpha float64, a []float64, lda uintptr, x []float64, beta float64, y []float64) bool {
	return false
}
