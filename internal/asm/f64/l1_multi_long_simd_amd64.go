// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
)

// l1NormLongSIMD is called only for native AVX2-capable lengths in [128, 1<<30].
// The result guard preserves the original portable grouping outside a domain
// in which both positive-addition graphs are guaranteed to remain finite.
func l1NormLongSIMD(x []float64) float64 {
	a := simd.BroadcastFloat64s(0)
	width := a.Len()
	if width < 1 || width > 8 {
		return l1NormPortableSIMD(x)
	}
	b := a
	c := a
	d := a
	step := 4 * width
	var i int
	for ; i <= len(x)-step; i += step {
		a = simd.LoadFloat64s(x[i : i+width]).Abs().Add(a)
		b = simd.LoadFloat64s(x[i+1*width : i+2*width]).Abs().Add(b)
		c = simd.LoadFloat64s(x[i+2*width : i+3*width]).Abs().Add(c)
		d = simd.LoadFloat64s(x[i+3*width : i+4*width]).Abs().Add(d)
	}
	// Finish complete vectors in the partial block before reducing lanes.
	if i <= len(x)-width {
		a = simd.LoadFloat64s(x[i : i+width]).Abs().Add(a)
		i += width
	}
	if i <= len(x)-width {
		b = simd.LoadFloat64s(x[i : i+width]).Abs().Add(b)
		i += width
	}
	if i <= len(x)-width {
		c = simd.LoadFloat64s(x[i : i+width]).Abs().Add(c)
		i += width
	}
	sum := reduceF64(a.Add(b).Add(c.Add(d)))
	for ; i < len(x); i++ {
		sum += math.Abs(x[i])
	}
	if sum <= math.MaxFloat64/2 {
		return sum
	}
	return l1NormPortableSIMD(x)
}
