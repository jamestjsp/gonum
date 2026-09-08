// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"simd"
	"simd/archsimd"
)

// Go initializes simd before this package. Its width and emulation settings,
// like the runtime CPU feature flags, remain fixed after startup. A zero limit
// disables the native entry; a positive limit also encodes its length bound.
var nativeShortReductionLimitSIMD = func() int {
	if !simd.Emulated() && archsimd.X86.AVX() {
		return 32
	}
	return 0
}()

// Keep the medium admission independent of AVX: startup overrides can disable
// AVX or AVX2 separately while native512 remains available.
var nativeMediumSumLimitSIMD = func() int {
	if !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() {
		return 128
	}
	return 0
}()
