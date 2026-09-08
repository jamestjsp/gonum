// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func gerEightHardwareSIMD(m uintptr, alpha float64, x, y, a []float64, lda uintptr) bool {
	return false
}
