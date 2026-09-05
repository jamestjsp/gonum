// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"simd"
	"unsafe"
)

const (
	dgemmSIMDRows = 4
	dgemmSIMDCols = 4
)

// dgemmSerialSIMD keeps a four-row, four-vector C tile in registers across k.
// The portable loop structure also appears in Go's SIMD GEMM experiments:
// https://go-review.googlesource.com/c/go/+/827812
func dgemmSerialSIMD(aTrans, bTrans bool, m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, alpha float64) bool {
	if bTrans || m < dgemmSIMDRows || k < 4 || n < 4 || !gemmSIMDDisjoint(c, a) || !gemmSIMDDisjoint(c, b) {
		return false
	}
	width := simd.BroadcastFloat64s(0).Len()
	tile := dgemmSIMDCols * width
	if n < tile {
		return false
	}
	rowStride, kStride := lda, 1
	if aTrans {
		rowStride, kStride = 1, lda
	}
	rows := m - m%dgemmSIMDRows
	for i := 0; i < rows; i += dgemmSIMDRows {
		a0 := a[(i+0)*rowStride : (i+0)*rowStride+(k-1)*kStride+1]
		c0 := c[(i+0)*ldc : (i+0)*ldc+n]
		a1 := a[(i+1)*rowStride : (i+1)*rowStride+(k-1)*kStride+1]
		c1 := c[(i+1)*ldc : (i+1)*ldc+n]
		a2 := a[(i+2)*rowStride : (i+2)*rowStride+(k-1)*kStride+1]
		c2 := c[(i+2)*ldc : (i+2)*ldc+n]
		a3 := a[(i+3)*rowStride : (i+3)*rowStride+(k-1)*kStride+1]
		c3 := c[(i+3)*ldc : (i+3)*ldc+n]
		j := 0
		for ; j+tile <= len(c0); j += tile {
			v00 := simd.LoadFloat64s(c0[j+0*width : j+1*width])
			v01 := simd.LoadFloat64s(c0[j+1*width : j+2*width])
			v02 := simd.LoadFloat64s(c0[j+2*width : j+3*width])
			v03 := simd.LoadFloat64s(c0[j+3*width : j+4*width])
			v10 := simd.LoadFloat64s(c1[j+0*width : j+1*width])
			v11 := simd.LoadFloat64s(c1[j+1*width : j+2*width])
			v12 := simd.LoadFloat64s(c1[j+2*width : j+3*width])
			v13 := simd.LoadFloat64s(c1[j+3*width : j+4*width])
			v20 := simd.LoadFloat64s(c2[j+0*width : j+1*width])
			v21 := simd.LoadFloat64s(c2[j+1*width : j+2*width])
			v22 := simd.LoadFloat64s(c2[j+2*width : j+3*width])
			v23 := simd.LoadFloat64s(c2[j+3*width : j+4*width])
			v30 := simd.LoadFloat64s(c3[j+0*width : j+1*width])
			v31 := simd.LoadFloat64s(c3[j+1*width : j+2*width])
			v32 := simd.LoadFloat64s(c3[j+2*width : j+3*width])
			v33 := simd.LoadFloat64s(c3[j+3*width : j+4*width])
			ai, bp := 0, j
			for l := 0; l < k; l++ {
				bv := b[bp : bp+tile]
				b0 := simd.LoadFloat64s(bv[0*width : 1*width])
				b1 := simd.LoadFloat64s(bv[1*width : 2*width])
				b2 := simd.LoadFloat64s(bv[2*width : 3*width])
				b3 := simd.LoadFloat64s(bv[3*width : 4*width])
				if scale := alpha * a0[ai]; scale != 0 {
					av := simd.BroadcastFloat64s(scale)
					v00 = av.MulAdd(b0, v00)
					v01 = av.MulAdd(b1, v01)
					v02 = av.MulAdd(b2, v02)
					v03 = av.MulAdd(b3, v03)
				}
				if scale := alpha * a1[ai]; scale != 0 {
					av := simd.BroadcastFloat64s(scale)
					v10 = av.MulAdd(b0, v10)
					v11 = av.MulAdd(b1, v11)
					v12 = av.MulAdd(b2, v12)
					v13 = av.MulAdd(b3, v13)
				}
				if scale := alpha * a2[ai]; scale != 0 {
					av := simd.BroadcastFloat64s(scale)
					v20 = av.MulAdd(b0, v20)
					v21 = av.MulAdd(b1, v21)
					v22 = av.MulAdd(b2, v22)
					v23 = av.MulAdd(b3, v23)
				}
				if scale := alpha * a3[ai]; scale != 0 {
					av := simd.BroadcastFloat64s(scale)
					v30 = av.MulAdd(b0, v30)
					v31 = av.MulAdd(b1, v31)
					v32 = av.MulAdd(b2, v32)
					v33 = av.MulAdd(b3, v33)
				}
				ai += kStride
				bp += ldb
			}
			v00.Store(c0[j+0*width : j+1*width])
			v01.Store(c0[j+1*width : j+2*width])
			v02.Store(c0[j+2*width : j+3*width])
			v03.Store(c0[j+3*width : j+4*width])
			v10.Store(c1[j+0*width : j+1*width])
			v11.Store(c1[j+1*width : j+2*width])
			v12.Store(c1[j+2*width : j+3*width])
			v13.Store(c1[j+3*width : j+4*width])
			v20.Store(c2[j+0*width : j+1*width])
			v21.Store(c2[j+1*width : j+2*width])
			v22.Store(c2[j+2*width : j+3*width])
			v23.Store(c2[j+3*width : j+4*width])
			v30.Store(c3[j+0*width : j+1*width])
			v31.Store(c3[j+1*width : j+2*width])
			v32.Store(c3[j+2*width : j+3*width])
			v33.Store(c3[j+3*width : j+4*width])
		}
		for ; j+width <= len(c0); j += width {
			v0 := simd.LoadFloat64s(c0[j : j+width])
			v1 := simd.LoadFloat64s(c1[j : j+width])
			v2 := simd.LoadFloat64s(c2[j : j+width])
			v3 := simd.LoadFloat64s(c3[j : j+width])
			ai, bp := 0, j
			for l := 0; l < k; l++ {
				bv := simd.LoadFloat64s(b[bp : bp+width])
				if scale := alpha * a0[ai]; scale != 0 {
					v0 = simd.BroadcastFloat64s(scale).MulAdd(bv, v0)
				}
				if scale := alpha * a1[ai]; scale != 0 {
					v1 = simd.BroadcastFloat64s(scale).MulAdd(bv, v1)
				}
				if scale := alpha * a2[ai]; scale != 0 {
					v2 = simd.BroadcastFloat64s(scale).MulAdd(bv, v2)
				}
				if scale := alpha * a3[ai]; scale != 0 {
					v3 = simd.BroadcastFloat64s(scale).MulAdd(bv, v3)
				}
				ai += kStride
				bp += ldb
			}
			v0.Store(c0[j : j+width])
			v1.Store(c1[j : j+width])
			v2.Store(c2[j : j+width])
			v3.Store(c3[j : j+width])
		}
		for ; j < len(c0); j++ {
			v0, v1, v2, v3 := c0[j], c1[j], c2[j], c3[j]
			ai, bp := 0, j
			for l := 0; l < k; l++ {
				bv := b[bp]
				if scale := alpha * a0[ai]; scale != 0 {
					v0 += scale * bv
				}
				if scale := alpha * a1[ai]; scale != 0 {
					v1 += scale * bv
				}
				if scale := alpha * a2[ai]; scale != 0 {
					v2 += scale * bv
				}
				if scale := alpha * a3[ai]; scale != 0 {
					v3 += scale * bv
				}
				ai += kStride
				bp += ldb
			}
			c0[j], c1[j], c2[j], c3[j] = v0, v1, v2, v3
		}
	}
	if rows < m {
		if aTrans {
			dgemmSerialTransNot(m-rows, n, k, a[rows:], lda, b, ldb, c[rows*ldc:], ldc, alpha)
		} else {
			dgemmSerialNotNot(m-rows, n, k, a[rows*lda:], lda, b, ldb, c[rows*ldc:], ldc, alpha)
		}
	}
	return true
}

func gemmSIMDDisjoint(x, y []float64) bool {
	xp, yp := uintptr(unsafe.Pointer(unsafe.SliceData(x))), uintptr(unsafe.Pointer(unsafe.SliceData(y)))
	const size = unsafe.Sizeof(float64(0))
	return xp+uintptr(len(x))*size <= yp || yp+uintptr(len(y))*size <= xp
}
