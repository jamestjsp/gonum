// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd"
)

func dotcUnitaryUncachedReference(x, y []complex64) complex64 {
	n := len(x)
	if n >= 4 && complexNativeSIMD() {
		width := simd.VectorBitSize()
		if width >= 256 && (width == 256 || n < 512) {
			// Call the native leaf directly from this public entry, avoiding
			// the shared helper's additional argument-copy frame.
			sum := complexDotShortWideSIMD(x, y, true)
			if math.Float32bits(real(sum))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(sum))&0x7f800000 != 0x7f800000 {
				return sum
			}
		}
	}
	// Preserve the complete established grouping and exceptional recovery.
	return portableDotUnitarySIMD(x, y, true)
}

func dotuUnitaryUncachedReference(x, y []complex64) complex64 {
	n := len(x)
	if n >= 4 && complexNativeSIMD() {
		width := simd.VectorBitSize()
		if width >= 256 && (width == 256 || n < 512) {
			// Call the native leaf directly from this public entry, avoiding
			// the shared helper's additional argument-copy frame.
			sum := complexDotShortWideSIMD(x, y, false)
			if math.Float32bits(real(sum))&0x7f800000 != 0x7f800000 && math.Float32bits(imag(sum))&0x7f800000 != 0x7f800000 {
				return sum
			}
		}
	}
	// Preserve the complete established grouping and exceptional recovery.
	return portableDotUnitarySIMD(x, y, false)
}
