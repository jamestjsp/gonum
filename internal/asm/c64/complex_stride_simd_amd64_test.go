// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"simd/archsimd"
	"testing"
)

// Exercise this fallback on AVX2 hosts with GODEBUG=simd=128,cpu.avx2=off.
// Portable SIMD128 needs only AVX, while the native leaves also use AVX2.
func TestSIMDComplexNativeRequiresAVX2(t *testing.T) {
	if archsimd.X86.AVX2() {
		t.Skip("AVX2 is available")
	}
	if complexNativeSIMD() {
		t.Fatal("native complex kernels selected without AVX2")
	}
}
