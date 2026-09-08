// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import "simd"

// The imported simd package and CPU features are initialized before c64.
// Their native eligibility and width remain fixed after startup.
// An inclusive bound preserves every valid slice length at width256.
var nativeDotUnitaryMaxSIMD = func() int {
	if !complexNativeSIMD() {
		return -1
	}
	width := simd.VectorBitSize()
	if width < 256 {
		return -1
	}
	if width == 256 {
		return int(^uint(0) >> 1)
	}
	return 511
}()
