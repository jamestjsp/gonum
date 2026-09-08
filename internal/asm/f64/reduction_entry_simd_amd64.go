// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

func DotUnitarySIMD(x, y []float64) float64 {
	if len(x) >= nativeShortReductionLimitSIMD {
		return dotUnitaryPortableEntrySIMD(x, y)
	}

	y = y[:len(x):len(x)]
	n := len(x)
	xp := unsafe.Pointer(unsafe.SliceData(x))
	yp := unsafe.Pointer(unsafe.SliceData(y))
	var a, b, c, d archsimd.Float64x2
	for n >= 8 {
		xb := (*[8]float64)(xp)
		yb := (*[8]float64)(yp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[0:2]))).Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[2:4]))).Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[4:6]))).Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yb[6:8]))).Add(d)
		n -= 8
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 64)
		yp = unsafe.Add(yp, 64)
	}
	if n >= 2 {
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(yp))).Add(a)
	}
	if n >= 4 {
		b = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 16))).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 16)))).Add(b)
	}
	if n >= 6 {
		c = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 32))).Mul(archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(yp, 32)))).Add(c)
	}
	a = a.Add(b).Add(c.Add(d))
	result := a.GetElem(0) + a.GetElem(1)
	if n&1 != 0 {
		result += *(*float64)(unsafe.Add(xp, (n-1)*8)) * *(*float64)(unsafe.Add(yp, (n-1)*8))
	}
	if math.Float64bits(result)&0x7ff0000000000000 != 0x7ff0000000000000 {
		return result
	}
	return dotUnitaryPortableEntrySIMD(x, y)
}
func SumSIMD(x []float64) float64 {
	if len(x) >= nativeShortReductionLimitSIMD {
		if len(x) < nativeMediumSumLimitSIMD {
			n := len(x)
			xp := unsafe.Pointer(unsafe.SliceData(x))
			var a, b, c, d archsimd.Float64x4
			for n >= 16 {
				v := (*[16]float64)(xp)
				a = archsimd.LoadFloat64x4Array((*[4]float64)(v[:4])).Add(a)
				b = archsimd.LoadFloat64x4Array((*[4]float64)(v[4:8])).Add(b)
				c = archsimd.LoadFloat64x4Array((*[4]float64)(v[8:12])).Add(c)
				d = archsimd.LoadFloat64x4Array((*[4]float64)(v[12:16])).Add(d)
				n -= 16
				if n == 0 {
					break
				}
				xp = unsafe.Add(xp, 128)
			}
			if n >= 4 {
				a = archsimd.LoadFloat64x4Array((*[4]float64)(xp)).Add(a)
			}
			if n >= 8 {
				b = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, 32))).Add(b)
			}
			if n >= 12 {
				c = archsimd.LoadFloat64x4Array((*[4]float64)(unsafe.Add(xp, 64))).Add(c)
			}
			a = a.Add(b).Add(c.Add(d))
			pair := a.GetLo().Add(a.GetHi())
			rem := n & 3
			if rem > 0 {
				tp := unsafe.Add(xp, (n-rem)*8)
				if rem >= 2 {
					pair = pair.Add(archsimd.LoadFloat64x2Array((*[2]float64)(tp)))
				}
				if rem&1 != 0 {
					var tail archsimd.Uint64x2
					tail = tail.SetElem(0, *(*uint64)(unsafe.Add(tp, (rem-1)*8)))
					pair = pair.Add(tail.AsFloat64x2())
				}
			}
			var reduced [2]uint64
			pair.ConcatAddPairs(pair).ToBits().StoreArray(&reduced)
			archsimd.ClearAVXUpperBits()
			bits := reduced[0]
			if bits&0x7ff0000000000000 != 0x7ff0000000000000 {
				return math.Float64frombits(bits)
			}
		}
		return sumPortableEntrySIMD(x)
	}

	n := len(x)
	xp := unsafe.Pointer(unsafe.SliceData(x))
	var a, b, c, d archsimd.Float64x2
	for n >= 16 {
		xb := (*[16]float64)(xp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Add(d)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[8:10])).Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[10:12])).Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[12:14])).Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[14:16])).Add(d)
		n -= 16
		if n == 0 {
			break
		}
		xp = unsafe.Add(xp, 128)
	}
	if n >= 8 {
		xb := (*[8]float64)(xp)
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xb[0:2])).Add(a)
		b = archsimd.LoadFloat64x2Array((*[2]float64)(xb[2:4])).Add(b)
		c = archsimd.LoadFloat64x2Array((*[2]float64)(xb[4:6])).Add(c)
		d = archsimd.LoadFloat64x2Array((*[2]float64)(xb[6:8])).Add(d)
		n -= 8
		if n > 0 {
			xp = unsafe.Add(xp, 64)
		}
	}
	if n >= 2 {
		a = archsimd.LoadFloat64x2Array((*[2]float64)(xp)).Add(a)
	}
	if n >= 4 {
		b = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 16))).Add(b)
	}
	if n >= 6 {
		c = archsimd.LoadFloat64x2Array((*[2]float64)(unsafe.Add(xp, 32))).Add(c)
	}
	a = a.Add(b).Add(c.Add(d))
	result := a.GetElem(0) + a.GetElem(1)
	if n&1 != 0 {
		result += *(*float64)(unsafe.Add(xp, (n-1)*8))
	}
	if math.Float64bits(result)&0x7ff0000000000000 != 0x7ff0000000000000 {
		return result
	}
	return sumPortableEntrySIMD(x)
}
