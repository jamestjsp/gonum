// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd/archsimd"
	"unsafe"
)

// A complete eight-column row fits one native 256-bit vector. Reuse y across
// all rows instead of entering the wide kernel's per-block narrow tail helper.
// The caller establishes native256 or wider, and disjoint matrix/input slices.
func gerEightHardwareSIMD(m uintptr, alpha float32, x []float32, incX, ix uintptr, y, a []float32, lda uintptr) {
	alphaV := archsimd.BroadcastFloat32x8(alpha)
	yv := archsimd.LoadFloat32x8Array((*[8]float32)(y[:8]))
	for row := uintptr(0); row < m; row++ {
		av := a[row*lda : row*lda+8]
		// Integer broadcast avoids a legacy scalar SSE load/multiply between
		// AVX rows; multiply before y to retain the original rounding order.
		scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Pointer(&x[ix]))).AsFloat32x8().Mul(alphaV)
		yv.Mul(scale).Add(archsimd.LoadFloat32x8Array((*[8]float32)(av))).StoreArray((*[8]float32)(av))
		ix += incX
	}
}
