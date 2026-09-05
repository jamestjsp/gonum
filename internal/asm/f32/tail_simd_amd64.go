// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import "simd/archsimd"

// Go 1.27.1 partial loads can cross a slice boundary. Complete 128-bit chunks
// bound every memory access and leave at most three scalar elements.
func axpyTailSIMD(dst []float32, alpha float32, x, y []float32) {
	a := archsimd.BroadcastFloat32x4(alpha)
	archsimd.LoadFloat32x4Array((*[4]float32)(x[:4])).Mul(a).Add(archsimd.LoadFloat32x4Array((*[4]float32)(y[:4]))).StoreArray((*[4]float32)(dst[:4]))
}

func gerTailSIMD(a0, a1, a2, a3, y []float32, s0, s1, s2, s3 float32) int {
	n := len(y) / 4 * 4
	y, a0, a1, a2, a3 = y[:n:n], a0[:n:n], a1[:n:n], a2[:n:n], a3[:n:n]
	x0, x1 := archsimd.BroadcastFloat32x4(s0), archsimd.BroadcastFloat32x4(s1)
	x2, x3 := archsimd.BroadcastFloat32x4(s2), archsimd.BroadcastFloat32x4(s3)
	for i := 0; i < n; i += 4 {
		v := archsimd.LoadFloat32x4Array((*[4]float32)(y[i : i+4]))
		v.Mul(x0).Add(archsimd.LoadFloat32x4Array((*[4]float32)(a0[i : i+4]))).StoreArray((*[4]float32)(a0[i : i+4]))
		v.Mul(x1).Add(archsimd.LoadFloat32x4Array((*[4]float32)(a1[i : i+4]))).StoreArray((*[4]float32)(a1[i : i+4]))
		v.Mul(x2).Add(archsimd.LoadFloat32x4Array((*[4]float32)(a2[i : i+4]))).StoreArray((*[4]float32)(a2[i : i+4]))
		v.Mul(x3).Add(archsimd.LoadFloat32x4Array((*[4]float32)(a3[i : i+4]))).StoreArray((*[4]float32)(a3[i : i+4]))
	}
	return n
}
