// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && simdbenchclean && !safe && !noasm && !gccgo

package simdbench

import "simd/archsimd"

// Apply the same timed state preparation before both implementations. A single
// clear before the whole trial does not guarantee stable upper-vector state.
// This inline helper leaves the original kernels and production dispatch intact.
func clearBenchmarkAVXState() {
	if archsimd.X86.AVX() {
		archsimd.ClearAVXUpperBits()
	}
}
