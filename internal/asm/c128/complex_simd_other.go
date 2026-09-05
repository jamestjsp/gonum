// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import "simd"

func complexSwapSIMD(x simd.Float64s) simd.Float64s {
	return complexSwapPortableSIMD(x)
}

func complexAxpySIMD(dst, x, y []float64, alpha complex128) int       { return -1 }
func complexScalSIMD(x []float64, alpha complex128) int               { return -1 }
func complexDotSIMD(x, y []float64, conjugate bool) (complex128, int) { return 0, -1 }
