// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

func gerEightHardwareSIMD(m uintptr, alpha float32, x []float32, incX, ix uintptr, y, a []float32, lda uintptr) {
	panic("unreachable native AMD64 kernel")
}
