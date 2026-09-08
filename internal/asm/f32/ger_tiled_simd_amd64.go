// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd/archsimd"
	"unsafe"
)

// Reuse up to 64 gathered y values across a 64-row block. Smaller register
// tiles and a shared remainder row loop preserve bounded memory operations.
func gerTiledPositiveSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) {
	xp, yp, ap := unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0]), unsafe.Pointer(&a[0])
	incX, incY, lda = incX*4, incY*4, lda*4
	alphaV := archsimd.BroadcastFloat32x8(alpha)
	for first := uintptr(0); first < m; {
		rows := min(uintptr(64), m-first)
		xb, ab := unsafe.Add(xp, first*incX), unsafe.Add(ap, first*lda)
		j := uintptr(0)
		for ; j+64 <= n; j += 64 {
			y0 := gatherGer8(unsafe.Add(yp, (j+0)*incY), incY)
			y1 := gatherGer8(unsafe.Add(yp, (j+8)*incY), incY)
			y2 := gatherGer8(unsafe.Add(yp, (j+16)*incY), incY)
			y3 := gatherGer8(unsafe.Add(yp, (j+24)*incY), incY)
			y4 := gatherGer8(unsafe.Add(yp, (j+32)*incY), incY)
			y5 := gatherGer8(unsafe.Add(yp, (j+40)*incY), incY)
			y6 := gatherGer8(unsafe.Add(yp, (j+48)*incY), incY)
			y7 := gatherGer8(unsafe.Add(yp, (j+56)*incY), incY)
			for i := uintptr(0); i < rows; i++ {
				scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, i*incX))).AsFloat32x8().Mul(alphaV)
				row := unsafe.Add(ab, i*lda+j*4)
				a0 := (*[8]float32)(unsafe.Add(row, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat32x8Array(a0)).StoreArray(a0)
				a1 := (*[8]float32)(unsafe.Add(row, 32))
				y1.Mul(scale).Add(archsimd.LoadFloat32x8Array(a1)).StoreArray(a1)
				a2 := (*[8]float32)(unsafe.Add(row, 64))
				y2.Mul(scale).Add(archsimd.LoadFloat32x8Array(a2)).StoreArray(a2)
				a3 := (*[8]float32)(unsafe.Add(row, 96))
				y3.Mul(scale).Add(archsimd.LoadFloat32x8Array(a3)).StoreArray(a3)
				a4 := (*[8]float32)(unsafe.Add(row, 128))
				y4.Mul(scale).Add(archsimd.LoadFloat32x8Array(a4)).StoreArray(a4)
				a5 := (*[8]float32)(unsafe.Add(row, 160))
				y5.Mul(scale).Add(archsimd.LoadFloat32x8Array(a5)).StoreArray(a5)
				a6 := (*[8]float32)(unsafe.Add(row, 192))
				y6.Mul(scale).Add(archsimd.LoadFloat32x8Array(a6)).StoreArray(a6)
				a7 := (*[8]float32)(unsafe.Add(row, 224))
				y7.Mul(scale).Add(archsimd.LoadFloat32x8Array(a7)).StoreArray(a7)
			}
		}
		for ; j+32 <= n; j += 32 {
			y0 := gatherGer8(unsafe.Add(yp, (j+0)*incY), incY)
			y1 := gatherGer8(unsafe.Add(yp, (j+8)*incY), incY)
			y2 := gatherGer8(unsafe.Add(yp, (j+16)*incY), incY)
			y3 := gatherGer8(unsafe.Add(yp, (j+24)*incY), incY)
			for i := uintptr(0); i < rows; i++ {
				scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, i*incX))).AsFloat32x8().Mul(alphaV)
				row := unsafe.Add(ab, i*lda+j*4)
				a0 := (*[8]float32)(unsafe.Add(row, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat32x8Array(a0)).StoreArray(a0)
				a1 := (*[8]float32)(unsafe.Add(row, 32))
				y1.Mul(scale).Add(archsimd.LoadFloat32x8Array(a1)).StoreArray(a1)
				a2 := (*[8]float32)(unsafe.Add(row, 64))
				y2.Mul(scale).Add(archsimd.LoadFloat32x8Array(a2)).StoreArray(a2)
				a3 := (*[8]float32)(unsafe.Add(row, 96))
				y3.Mul(scale).Add(archsimd.LoadFloat32x8Array(a3)).StoreArray(a3)
			}
		}
		for ; j+16 <= n; j += 16 {
			y0 := gatherGer8(unsafe.Add(yp, (j+0)*incY), incY)
			y1 := gatherGer8(unsafe.Add(yp, (j+8)*incY), incY)
			for i := uintptr(0); i < rows; i++ {
				scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, i*incX))).AsFloat32x8().Mul(alphaV)
				row := unsafe.Add(ab, i*lda+j*4)
				a0 := (*[8]float32)(unsafe.Add(row, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat32x8Array(a0)).StoreArray(a0)
				a1 := (*[8]float32)(unsafe.Add(row, 32))
				y1.Mul(scale).Add(archsimd.LoadFloat32x8Array(a1)).StoreArray(a1)
			}
		}
		if j < n {
			remain := n - j
			if tail := remain & 7; tail >= 5 {
				// The final full window ends at n. Its first 8-tail lanes
				// belong to an earlier block; preserve that block's result.
				last := n - 8
				yLast := gatherGer8(unsafe.Add(yp, last*incY), incY)
				var y8 archsimd.Float32x8
				if remain >= 8 {
					y8 = gatherGer8(unsafe.Add(yp, j*incY), incY)
				}
				// Lane IDs [0,1,2,3,4,4,4,4] suffice for thresholds
				// 0..2. AVX2 comparison makes a register mask without
				// Mask32x8FromBits, whose API requires AVX512.
				var laneBits archsimd.Uint64x2
				laneBits = laneBits.SetElem(0, 0x0000000100000000).SetElem(1, 0x0000000300000002)
				laneIDs := archsimd.BroadcastInt32x8(4).SetLo(laneBits.AsInt32x4())
				active := laneIDs.Greater(archsimd.BroadcastInt32x8(int32(7 - tail)))
				for i := uintptr(0); i < rows; i++ {
					scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, i*incX))).AsFloat32x8().Mul(alphaV)
					row := unsafe.Add(ab, i*lda)
					dst := (*[8]float32)(unsafe.Add(row, last*4))
					before := archsimd.LoadFloat32x8Array(dst)
					result := yLast.Mul(scale).Add(before).IfElse(active, before)
					if remain >= 8 {
						first := (*[8]float32)(unsafe.Add(row, j*4))
						firstResult := y8.Mul(scale).Add(archsimd.LoadFloat32x8Array(first))
						// Read both windows before either store. Committing
						// the first block last avoids reloading its partial
						// overlap and updates those lanes exactly once.
						result.StoreArray(dst)
						firstResult.StoreArray(first)
					} else {
						result.StoreArray(dst)
					}
				}
			} else {
				tail := remain & 3
				j4 := j
				var y8 archsimd.Float32x8
				if remain >= 8 {
					y8 = gatherGer8(unsafe.Add(yp, j*incY), incY)
					j4 += 8
				}
				jt := j4
				var y4 archsimd.Float32x4
				if remain&4 != 0 {
					y4 = gatherStridedPointer4(unsafe.Add(yp, j4*incY), incY)
					jt += 4
				}
				var yt archsimd.Uint32x4
				if tail != 0 {
					yt = yt.SetElem(0, *(*uint32)(unsafe.Add(yp, jt*incY)))
				}
				if tail >= 2 {
					yt = yt.SetElem(1, *(*uint32)(unsafe.Add(yp, (jt+1)*incY)))
				}
				if tail == 3 {
					yt = yt.SetElem(2, *(*uint32)(unsafe.Add(yp, (jt+2)*incY)))
				}
				for i := uintptr(0); i < rows; i++ {
					scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, i*incX))).AsFloat32x8().Mul(alphaV)
					row := unsafe.Add(ab, i*lda)
					if remain >= 8 {
						dst := (*[8]float32)(unsafe.Add(row, j*4))
						y8.Mul(scale).Add(archsimd.LoadFloat32x8Array(dst)).StoreArray(dst)
					}
					if remain&4 != 0 {
						dst := (*[4]float32)(unsafe.Add(row, j4*4))
						y4.Mul(scale.GetLo()).Add(archsimd.LoadFloat32x4Array(dst)).StoreArray(dst)
					}
					if tail != 0 {
						dst := unsafe.Add(row, jt*4)
						var av archsimd.Uint32x4
						if tail >= 2 {
							var pair archsimd.Uint64x2
							av = pair.SetElem(0, *(*uint64)(dst)).AsUint32x4()
						} else {
							av = av.SetElem(0, *(*uint32)(dst))
						}
						if tail == 3 {
							av = av.SetElem(2, *(*uint32)(unsafe.Add(dst, 8)))
						}
						result := yt.AsFloat32x4().Mul(scale.GetLo()).Add(av.AsFloat32x4()).AsUint32x4()
						if tail >= 2 {
							*(*uint64)(dst) = result.AsUint64x2().GetElem(0)
						} else {
							*(*uint32)(dst) = result.GetElem(0)
						}
						if tail == 3 {
							*(*uint32)(unsafe.Add(dst, 8)) = result.GetElem(2)
						}
					}
				}
			}
		}
		first += rows
	}
}
