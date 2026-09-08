// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

// Bounds and independent inputs permit reusing each gathered column tile
// across rows. Sparse inputs load only their selected elements.
func gerTiledHardwareSIMD(m, n uintptr, alpha float64, x []float64, incX uintptr, y []float64, incY uintptr, a []float64, lda uintptr) bool {
	if m < 4 || n < 4 || n > uintptr(len(a)) || lda < n || !simdPositiveSpan(len(x), m, incX, 0) || !simdPositiveSpan(len(y), n, incY, 0) || m-1 > uintptr(len(a)-int(n))/lda || simd.Emulated() || simd.VectorBitSize() < 256 || !archsimd.X86.AVX2() || !simdMatrixDisjoint(a, x) || !simdMatrixDisjoint(a, y) {
		return false
	}
	if incY == 1 && n >= 16 && simd.VectorBitSize() >= 512 && archsimd.X86.AVX512() {
		gerTiledWideNativeSIMD(m, n, math.Float64bits(alpha), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(a)), incX*8, lda*8)
		return true
	}
	gerTiledNativeSIMD(m, n, math.Float64bits(alpha), unsafe.Pointer(unsafe.SliceData(x)), unsafe.Pointer(unsafe.SliceData(y)), unsafe.Pointer(unsafe.SliceData(a)), incX*8, incY*8, lda*8)
	return true
}

func gatherGerTwoSIMD(p unsafe.Pointer, inc uintptr) archsimd.Float64x2 {
	var v archsimd.Uint64x2
	return v.SetElem(0, *(*uint64)(p)).SetElem(1, *(*uint64)(unsafe.Add(p, inc))).AsFloat64x2()
}
func gatherGerFourSIMD(p unsafe.Pointer, inc uintptr) archsimd.Float64x4 {
	if inc == 8 {
		return archsimd.LoadFloat64x4Array((*[4]float64)(p))
	}
	var v archsimd.Float64x4
	return v.SetLo(gatherGerTwoSIMD(p, inc)).SetHi(gatherGerTwoSIMD(unsafe.Add(p, 2*inc), inc))
}

// Each destination element keeps the original two multiplications followed
// by an addition. All pointer expressions address validated live elements.
func gerTiledNativeSIMD(m, n uintptr, alphaBits uint64, xp, yp, ap unsafe.Pointer, incX, incY, lda uintptr) {
	alpha := archsimd.BroadcastUint64x4(alphaBits).AsFloat64x4()
	for first := uintptr(0); first < m; {
		rows := min(uintptr(64), m-first)
		xb, ab := unsafe.Add(xp, first*incX), unsafe.Add(ap, first*lda)
		j := uintptr(0)
		for ; j+16 <= n; j += 16 {
			y0 := gatherGerFourSIMD(unsafe.Add(yp, (j+0)*incY), incY)
			y1 := gatherGerFourSIMD(unsafe.Add(yp, (j+4)*incY), incY)
			y2 := gatherGerFourSIMD(unsafe.Add(yp, (j+8)*incY), incY)
			y3 := gatherGerFourSIMD(unsafe.Add(yp, (j+12)*incY), incY)
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x4(*(*uint64)(xr)).AsFloat64x4().Mul(alpha)
				row := ar
				a0 := (*[4]float64)(unsafe.Add(row, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat64x4Array(a0)).StoreArray(a0)
				a1 := (*[4]float64)(unsafe.Add(row, 32))
				y1.Mul(scale).Add(archsimd.LoadFloat64x4Array(a1)).StoreArray(a1)
				a2 := (*[4]float64)(unsafe.Add(row, 64))
				y2.Mul(scale).Add(archsimd.LoadFloat64x4Array(a2)).StoreArray(a2)
				a3 := (*[4]float64)(unsafe.Add(row, 96))
				y3.Mul(scale).Add(archsimd.LoadFloat64x4Array(a3)).StoreArray(a3)

				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		for ; j+8 <= n; j += 8 {
			y0 := gatherGerFourSIMD(unsafe.Add(yp, (j+0)*incY), incY)
			y1 := gatherGerFourSIMD(unsafe.Add(yp, (j+4)*incY), incY)
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x4(*(*uint64)(xr)).AsFloat64x4().Mul(alpha)
				row := ar
				a0 := (*[4]float64)(unsafe.Add(row, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat64x4Array(a0)).StoreArray(a0)
				a1 := (*[4]float64)(unsafe.Add(row, 32))
				y1.Mul(scale).Add(archsimd.LoadFloat64x4Array(a1)).StoreArray(a1)

				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		for ; j+4 <= n; j += 4 {
			y0 := gatherGerFourSIMD(unsafe.Add(yp, (j+0)*incY), incY)
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x4(*(*uint64)(xr)).AsFloat64x4().Mul(alpha)
				row := ar
				a0 := (*[4]float64)(unsafe.Add(row, 0))
				y0.Mul(scale).Add(archsimd.LoadFloat64x4Array(a0)).StoreArray(a0)

				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
		}
		if j+2 <= n {
			y2 := gatherGerTwoSIMD(unsafe.Add(yp, j*incY), incY)
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x2(*(*uint64)(xr)).AsFloat64x2().Mul(alpha.GetLo())
				dst := (*[2]float64)(ar)
				y2.Mul(scale).Add(archsimd.LoadFloat64x2Array(dst)).StoreArray(dst)

				left--
				if left == 0 {
					break
				}
				xr, ar = unsafe.Add(xr, incX), unsafe.Add(ar, lda)
			}
			j += 2
		}
		if j < n {
			y1 := archsimd.BroadcastUint64x2(*(*uint64)(unsafe.Add(yp, j*incY))).AsFloat64x2()
			xr, ar := xb, unsafe.Add(ab, j*8)
			for left := rows; left > 0; {
				scale := archsimd.BroadcastUint64x2(*(*uint64)(xr)).AsFloat64x2().Mul(alpha.GetLo())
				dst := (*uint64)(ar)
				value := archsimd.BroadcastUint64x2(*dst).AsFloat64x2()
				*dst = y1.Mul(scale).Add(value).AsUint64x2().GetElem(0)

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
