// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"simd"
	"simd/archsimd"
	"unsafe"
)

// Validate every addressed element once before entering the matrix kernel.
// Overlapping rows and inputs retain the sequential checked implementation.
func gerPositiveHardwareSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) bool {
	if !archsimd.X86.AVX2() || simd.VectorBitSize() < 256 || n > uintptr(len(a)) || lda < n || !positiveStrideSIMD(len(x), 0, m, incX) || !positiveStrideSIMD(len(y), 0, n, incY) || !positiveStrideSIMD(len(a)-int(n)+1, 0, m, lda) || !simdMatrixDisjoint(a, x) || !simdMatrixDisjoint(a, y) {
		return false
	}
	if m >= 8 && n >= 16 {
		gerTiledPositiveSIMD(m, n, alpha, x, incX, y, incY, a, lda)
	} else {
		gerStridedPositiveSIMD(m, n, alpha, x, incX, y, incY, a, lda)
	}
	return true
}

func gatherGer8(p unsafe.Pointer, inc uintptr) archsimd.Float32x8 {
	lo := gatherStridedPointer4(p, inc)
	hi := gatherStridedPointer4(unsafe.Add(p, 4*inc), inc)
	var value archsimd.Float32x8
	return value.SetLo(lo).SetHi(hi)
}

// Gather each y vector once for four rows. All arithmetic, including incomplete
// columns, uses AVX operations so row boundaries do not switch to legacy SSE.
func gerStridedPositiveSIMD(m, n uintptr, alpha float32, x []float32, incX uintptr, y []float32, incY uintptr, a []float32, lda uintptr) {
	xp, yp, ap := unsafe.Pointer(&x[0]), unsafe.Pointer(&y[0]), unsafe.Pointer(&a[0])
	incX, incY, lda = incX*4, incY*4, lda*4
	alphaV := archsimd.BroadcastFloat32x8(alpha)
	i := uintptr(0)
	for ; i+4 <= m; i += 4 {
		xb := unsafe.Add(xp, i*incX)
		s0 := archsimd.BroadcastUint32x8(*(*uint32)(xb)).AsFloat32x8().Mul(alphaV)
		s1 := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, incX))).AsFloat32x8().Mul(alphaV)
		s2 := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, 2*incX))).AsFloat32x8().Mul(alphaV)
		s3 := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xb, 3*incX))).AsFloat32x8().Mul(alphaV)
		a0 := unsafe.Add(ap, i*lda)
		a1, a2, a3 := unsafe.Add(a0, lda), unsafe.Add(a0, 2*lda), unsafe.Add(a0, 3*lda)
		j := uintptr(0)
		for ; j+8 <= n; j += 8 {
			v := gatherGer8(unsafe.Add(yp, j*incY), incY)
			r0, r1 := (*[8]float32)(unsafe.Add(a0, j*4)), (*[8]float32)(unsafe.Add(a1, j*4))
			r2, r3 := (*[8]float32)(unsafe.Add(a2, j*4)), (*[8]float32)(unsafe.Add(a3, j*4))
			v.Mul(s0).Add(archsimd.LoadFloat32x8Array(r0)).StoreArray(r0)
			v.Mul(s1).Add(archsimd.LoadFloat32x8Array(r1)).StoreArray(r1)
			v.Mul(s2).Add(archsimd.LoadFloat32x8Array(r2)).StoreArray(r2)
			v.Mul(s3).Add(archsimd.LoadFloat32x8Array(r3)).StoreArray(r3)
		}
		if j+4 <= n {
			v := gatherStridedPointer4(unsafe.Add(yp, j*incY), incY)
			r0, r1 := (*[4]float32)(unsafe.Add(a0, j*4)), (*[4]float32)(unsafe.Add(a1, j*4))
			r2, r3 := (*[4]float32)(unsafe.Add(a2, j*4)), (*[4]float32)(unsafe.Add(a3, j*4))
			v.Mul(s0.GetLo()).Add(archsimd.LoadFloat32x4Array(r0)).StoreArray(r0)
			v.Mul(s1.GetLo()).Add(archsimd.LoadFloat32x4Array(r1)).StoreArray(r1)
			v.Mul(s2.GetLo()).Add(archsimd.LoadFloat32x4Array(r2)).StoreArray(r2)
			v.Mul(s3.GetLo()).Add(archsimd.LoadFloat32x4Array(r3)).StoreArray(r3)
			j += 4
		}
		if j < n {
			scales := gatherStridedPointer4(xb, incX).Mul(alphaV.GetLo())
			for ; j < n; j++ {
				v := archsimd.BroadcastUint32x4(*(*uint32)(unsafe.Add(yp, j*incY))).AsFloat32x4()
				column := unsafe.Add(a0, j*4)
				result := v.Mul(scales).Add(gatherStridedPointer4(column, lda)).AsUint32x4()
				*(*uint32)(column) = result.GetElem(0)
				*(*uint32)(unsafe.Add(column, lda)) = result.GetElem(1)
				*(*uint32)(unsafe.Add(column, 2*lda)) = result.GetElem(2)
				*(*uint32)(unsafe.Add(column, 3*lda)) = result.GetElem(3)
			}
		}
	}
	for ; i < m; i++ {
		scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Add(xp, i*incX))).AsFloat32x8().Mul(alphaV)
		row := unsafe.Add(ap, i*lda)
		j := uintptr(0)
		for ; j+8 <= n; j += 8 {
			v := gatherGer8(unsafe.Add(yp, j*incY), incY)
			dst := (*[8]float32)(unsafe.Add(row, j*4))
			v.Mul(scale).Add(archsimd.LoadFloat32x8Array(dst)).StoreArray(dst)
		}
		if j+4 <= n {
			v := gatherStridedPointer4(unsafe.Add(yp, j*incY), incY)
			dst := (*[4]float32)(unsafe.Add(row, j*4))
			v.Mul(scale.GetLo()).Add(archsimd.LoadFloat32x4Array(dst)).StoreArray(dst)
			j += 4
		}
		for ; j < n; j++ {
			v := archsimd.BroadcastUint32x4(*(*uint32)(unsafe.Add(yp, j*incY))).AsFloat32x4()
			dst := (*uint32)(unsafe.Add(row, j*4))
			av := archsimd.BroadcastUint32x4(*dst).AsFloat32x4()
			*dst = v.Mul(scale.GetLo()).Add(av).AsUint32x4().GetElem(0)
		}
	}
}

// Eight columns share a single y vector across every row. Checked addresses
// avoid the full-span validation overhead for this small matrix shape.
func gerEightStridedHardwareSIMD(m uintptr, alpha float32, x []float32, incX, ix uintptr, y []float32, incY, iy uintptr, a []float32, lda uintptr) {
	alphaV := archsimd.BroadcastFloat32x8(alpha)
	var yv archsimd.Float32x8
	if incY == 1 {
		yv = archsimd.LoadFloat32x8Array((*[8]float32)(y[iy : iy+8]))
	} else {
		yv = yv.SetLo(gatherStrided4(y, iy, incY)).SetHi(gatherStrided4(y, iy+4*incY, incY))
	}
	for row := uintptr(0); row < m; row++ {
		av := a[row*lda : row*lda+8]
		scale := archsimd.BroadcastUint32x8(*(*uint32)(unsafe.Pointer(&x[ix]))).AsFloat32x8().Mul(alphaV)
		yv.Mul(scale).Add(archsimd.LoadFloat32x8Array((*[8]float32)(av))).StoreArray((*[8]float32)(av))
		ix += incX
	}
}
