// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"simd"
	"simd/archsimd"
	"unsafe"
)

func DotUnitarySIMD(x, y []float32) float32 {
	if len(x) >= nativeReductionLimitSIMD {
		return dotUnitaryPortableEntrySIMD(x, y)
	}
	var result float32
	if len(x) >= 32 {
		n := len(x)
		xp := unsafe.Pointer(&x[0])
		ys := y[:len(x):len(x)]
		yp := unsafe.Pointer(&ys[0])
		var a, b, c, d archsimd.Float32x8
		for n >= 32 {
			a = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 0))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 0)))).Add(a)
			b = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 32))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 32)))).Add(b)
			c = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 64))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 64)))).Add(c)
			d = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 96))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 96)))).Add(d)
			n -= 32
			if n == 0 {
				break
			}
			xp = unsafe.Add(xp, 128)
			yp = unsafe.Add(yp, 128)
		}
		if n >= 8 {
			a = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 0))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 0)))).Add(a)
		}
		if n >= 16 {
			b = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 32))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 32)))).Add(b)
		}
		if n >= 24 {
			c = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 64))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 64)))).Add(c)
		}
		a = a.Add(b).Add(c.Add(d))
		quarter := a.GetLo().Add(a.GetHi())
		if n&7 >= 4 {
			quarter = quarter.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n&^7)*4))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(yp, (n&^7)*4)))))
		}
		if n&3 == 3 {
			tail := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n-4)*4))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(yp, (n-4)*4))))
			quarter = quarter.Add(tail.AsUint32x4().SetElem(0, 0).AsFloat32x4())
		}
		pair := quarter.Add(quarter.ConcatPermuteScalars(2, 3, 0, 1, quarter))
		// Order the reduction before clearing wide state. A pure lane extract
		// can otherwise move past the clear and introduce a legacy XMM spill.
		var lanes [4]float32
		pair.StoreArray(&lanes)
		archsimd.ClearAVXUpperBits()
		result = lanes[0] + lanes[1]
		if n&3 != 3 {
			for i := n &^ 3; i < n; i++ {
				result += *(*float32)(unsafe.Add(xp, i*4)) * *(*float32)(unsafe.Add(yp, i*4))
			}
		}
		goto finish
	}
	// Pack the previous four 128-bit accumulators into two wider vectors.
	// Normalize both halves of low to +0. The first accumulator cannot be -0,
	// so normalizing the second chunk does not change their eventual a+b fold.
	if len(x) >= 16 && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() {
		n := len(x)
		xp := unsafe.Pointer(&x[0])
		ys := y[:len(x):len(x)]
		yp := unsafe.Pointer(&ys[0])
		low := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 0))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 0))))
		// Derive +0 from admitted loads so the compiler cannot hoist a wide
		// zero initializer above the AVX2 check. Nonfinite products enter the
		// unchanged complete recovery path below.
		low = low.Add(low.Sub(low))
		high := archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 32))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 32))))
		if n >= 24 {
			low = low.Add(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 64))).Mul(archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(yp, 64)))))
		}
		a, b := low.GetLo(), low.GetHi()
		c, d := high.GetLo(), high.GetHi()
		if n&7 >= 4 {
			c = c.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n&^7)*4))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(yp, (n&^7)*4)))))
		}
		if n&3 == 3 {
			tail := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n-4)*4))).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(yp, (n-4)*4))))
			d = d.Add(tail.AsUint32x4().SetElem(0, 0).AsFloat32x4())
		}
		acc := a.Add(b).Add(c.Add(d))
		pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
		var lanes [4]float32
		pair.StoreArray(&lanes)
		archsimd.ClearAVXUpperBits()
		result = lanes[0] + lanes[1]
		if n&3 != 3 {
			for i := n &^ 3; i < n; i++ {
				result += *(*float32)(unsafe.Add(xp, i*4)) * *(*float32)(unsafe.Add(yp, i*4))
			}
		}
		goto finish
	}
	{
		xs, ys := x, y

		ys = ys[:len(xs):len(xs)]
		if len(xs) < 4 {
			var sum float32
			for i, v := range xs {
				sum += v * ys[i]
			}
			result = sum
			goto finish
		}
		lastX, lastY := xs[len(xs)-4:], ys[len(xs)-4:]
		var a, b, c, d archsimd.Float32x4
		if len(xs) >= 16 {
			a = archsimd.LoadFloat32x4Array((*[4]float32)(xs[:4])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[:4]))).Add(a)
			b = archsimd.LoadFloat32x4Array((*[4]float32)(xs[4:8])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[4:8])))
			c = archsimd.LoadFloat32x4Array((*[4]float32)(xs[8:12])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[8:12])))
			d = archsimd.LoadFloat32x4Array((*[4]float32)(xs[12:16])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[12:16])))
			xs, ys = xs[16:], ys[16:]
		}
		if len(xs) >= 8 {
			a = archsimd.LoadFloat32x4Array((*[4]float32)(xs[:4])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[:4]))).Add(a)
			b = archsimd.LoadFloat32x4Array((*[4]float32)(xs[4:8])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[4:8]))).Add(b)
			xs, ys = xs[8:], ys[8:]
		}
		if len(xs) >= 4 {
			c = archsimd.LoadFloat32x4Array((*[4]float32)(xs[:4])).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(ys[:4]))).Add(c)
			xs, ys = xs[4:], ys[4:]
		}
		if len(xs) == 3 {
			tail := archsimd.LoadFloat32x4Array((*[4]float32)(lastX)).Mul(archsimd.LoadFloat32x4Array((*[4]float32)(lastY)))
			d = d.Add(tail.AsUint32x4().SetElem(0, 0).AsFloat32x4())
			acc := a.Add(b).Add(c.Add(d))
			pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
			result = pair.GetElem(0) + pair.GetElem(1)
			goto finish
		}
		acc := a.Add(b).Add(c.Add(d))
		pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
		sum := pair.GetElem(0) + pair.GetElem(1)
		for i, v := range xs {
			sum += v * ys[i]
		}
		result = sum
		goto finish

	}
finish:
	if math.Float32bits(result)&0x7f800000 == 0x7f800000 {
		result = dotUnitaryOriginalSIMD(x, y)
		if math.Float32bits(result)&0x7f800000 == 0x7f800000 {
			return dotIncSequentialSIMD(x, y, uintptr(len(x)), 1, 1, 0, 0)
		}
	}
	return result
}

func SumSIMD(x []float32) float32 {
	if len(x) >= nativeReductionLimitSIMD {
		return sumPortableEntrySIMD(x)
	}
	var result float32
	if len(x) >= 32 {
		n := len(x)
		xp := unsafe.Pointer(&x[0])
		var a, b, c, d archsimd.Float32x8
		for n >= 32 {
			a = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 0))).Add(a)
			b = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 32))).Add(b)
			c = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 64))).Add(c)
			d = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 96))).Add(d)
			n -= 32
			if n == 0 {
				break
			}
			xp = unsafe.Add(xp, 128)
		}
		if n >= 8 {
			a = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 0))).Add(a)
		}
		if n >= 16 {
			b = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 32))).Add(b)
		}
		if n >= 24 {
			c = archsimd.LoadFloat32x8Array((*[8]float32)(unsafe.Add(xp, 64))).Add(c)
		}
		a = a.Add(b).Add(c.Add(d))
		quarter := a.GetLo().Add(a.GetHi())
		if n&7 >= 4 {
			quarter = quarter.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n&^7)*4))))
		}
		if n&3 == 3 {
			tail := archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n-4)*4)))
			quarter = quarter.Add(tail.AsUint32x4().SetElem(0, 0).AsFloat32x4())
		}
		pair := quarter.Add(quarter.ConcatPermuteScalars(2, 3, 0, 1, quarter))
		// Complete the reduction store before clearing wide vector state.
		var lanes [4]float32
		pair.StoreArray(&lanes)
		archsimd.ClearAVXUpperBits()
		result = lanes[0] + lanes[1]
		if n&3 != 3 {
			for i := n &^ 3; i < n; i++ {
				result += *(*float32)(unsafe.Add(xp, i*4))
			}
		}
		goto finish
	}
	{
		xs := x

		if len(xs) < 4 {
			var sum float32
			for _, v := range xs {
				sum += v
			}
			result = sum
			goto finish
		}
		n := len(x)
		xp := unsafe.Pointer(&x[0])
		i := 0
		var a, b, c, d archsimd.Float32x4
		if n >= 16 {
			a = archsimd.LoadFloat32x4Array((*[4]float32)(xp)).Add(a)
			b = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, 16)))
			c = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, 32)))
			d = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, 48)))
			i = 16
		}
		if n-i >= 8 {
			a = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, i*4))).Add(a)
			b = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (i+4)*4))).Add(b)
			i += 8
		}
		if n-i >= 4 {
			c = archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, i*4))).Add(c)
			i += 4
		}
		if n-i == 3 {
			d = d.Add(archsimd.LoadFloat32x4Array((*[4]float32)(unsafe.Add(xp, (n-4)*4))).AsUint32x4().SetElem(0, 0).AsFloat32x4())
			acc := a.Add(b).Add(c.Add(d))
			pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
			result = pair.GetElem(0) + pair.GetElem(1)
			goto finish
		}
		acc := a.Add(b).Add(c.Add(d))
		pair := acc.Add(acc.ConcatPermuteScalars(2, 3, 0, 1, acc))
		sum := pair.GetElem(0) + pair.GetElem(1)
		for ; i < n; i++ {
			sum += *(*float32)(unsafe.Add(xp, i*4))
		}
		result = sum
		goto finish

	}

finish:
	if math.Float32bits(result)&0x7f800000 == 0x7f800000 {
		result = sumOriginalSIMD(x)
		if math.Float32bits(result)&0x7f800000 == 0x7f800000 {
			return sumSequentialSIMD(x)
		}
	}
	return result
}
