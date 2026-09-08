// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

func DotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float32 {
	return dotIncPortableEntrySIMD(x, y, n, incX, incY, ix, iy)
}
func DdotIncSIMD(x, y []float32, n, incX, incY, ix, iy uintptr) float64 {
	return ddotIncPortableEntrySIMD(x, y, n, incX, incY, ix, iy)
}
