// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"simd/archsimd"
)

// Native512 selection can remain enabled when AVX2 is disabled independently.
// The narrower broadcast helpers must check their own required CPU feature.
func supportsAVX2SIMD() bool { return archsimd.X86.AVX2() }

// Go initializes simd before this package. Its width and emulation settings,
// like the runtime CPU feature flags, remain fixed after startup. A zero limit
// disables the native entry; a positive limit also encodes its length bound.
var nativeReductionLimitSIMD = func() int {
	if simd.Emulated() || !archsimd.X86.AVX() {
		return 0
	}
	if simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() {
		return 256
	}
	return 32
}()

var nativeDdotShortLimitSIMD = func() int {
	if !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() {
		return 32
	}
	return 0
}()
