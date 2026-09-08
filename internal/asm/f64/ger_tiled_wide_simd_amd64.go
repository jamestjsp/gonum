// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"simd/archsimd"
	"unsafe"
)

// Contiguous y values use full-width vectors while sparse inputs retain the
// exact gather kernel. The caller validates both CPU features and every span.
func gerTiledWideNativeSIMD(m, n uintptr, alphaBits uint64, xp, yp, ap unsafe.Pointer, incX, lda uintptr) {
	alpha := archsimd.BroadcastUint64x8(alphaBits).AsFloat64x8()
	for first := uintptr(0); first < m; {
		rows := min(uintptr(64), m-first)
		xb, ab := unsafe.Add(xp, first*incX), unsafe.Add(ap, first*lda)
		j := uintptr(0)
		for ; j+32 <= n; j += 32 {
			y0 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+0)*8)))
			y1 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+8)*8)))
			y2 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+16)*8)))
			y3 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+24)*8)))
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x8(*(*uint64)(xr)).AsFloat64x8().Mul(alpha)
				dst0 := (*[8]float64)(unsafe.Add(ar, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst0)).StoreArray(dst0)
				dst1 := (*[8]float64)(unsafe.Add(ar, 64))
				y1.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst1)).StoreArray(dst1)
				dst2 := (*[8]float64)(unsafe.Add(ar, 128))
				y2.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst2)).StoreArray(dst2)
				dst3 := (*[8]float64)(unsafe.Add(ar, 192))
				y3.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst3)).StoreArray(dst3)
				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		for ; j+16 <= n; j += 16 {
			y0 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+0)*8)))
			y1 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+8)*8)))
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x8(*(*uint64)(xr)).AsFloat64x8().Mul(alpha)
				dst0 := (*[8]float64)(unsafe.Add(ar, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst0)).StoreArray(dst0)
				dst1 := (*[8]float64)(unsafe.Add(ar, 64))
				y1.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst1)).StoreArray(dst1)
				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		for ; j+8 <= n; j += 8 {
			y0 := archsimd.LoadFloat64x8Array((*[8]float64)(unsafe.Add(yp, (j+0)*8)))
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x8(*(*uint64)(xr)).AsFloat64x8().Mul(alpha)
				dst0 := (*[8]float64)(unsafe.Add(ar, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat64x8Array(dst0)).StoreArray(dst0)
				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		if j+4 <= n {
			y := archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(yp, j*8)))
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x4(*(*uint64)(xr)).AsFloat64x4().Mul(alpha.GetLo())
				dst := (*[4]float64)(ar)
				y.Mul(scale).Add(archsimd.LoadFloat64x4Array(dst)).StoreArray(dst)
				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
			j += 4
		}
		if j+2 <= n {
			y := archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, j*8)))
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x2(*(*uint64)(xr)).AsFloat64x2().Mul(alpha.GetLo().GetLo())
				dst := (*[2]float64)(ar)
				y.Mul(scale).Add(archsimd.LoadFloat64x2Array(dst)).StoreArray(dst)
				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
			j += 2
		}
		if j < n {
			y := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(yp, j*8))).AsFloat64x2()
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x2(*(*uint64)(xr)).AsFloat64x2().Mul(alpha.GetLo().GetLo())
				dst := (*uint64)(ar)
				value := archsimd.BroadcastUint64x2(*dst).AsFloat64x2()
				*dst = y.Mul(scale).Add(value).AsUint64x2().GetElem(0)
				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		first += rows
	}
	archsimd.ClearAVXUpperBits()
}
