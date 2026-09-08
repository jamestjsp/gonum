// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import "unsafe"

// These leaves run only after complete positive-span and disjointness checks.
// Keeping slice descriptors out of the leaf avoids spilling them in its loop.
func axpyIncPositiveSIMD(alpha float64, x, y unsafe.Pointer, n, sx, sy uintptr) {
	for n >= 8 {
		{
			v0 := alpha**simdStrideAt(x, 0*sx) + *simdStrideAt(y, 0*sy)
			v1 := alpha**simdStrideAt(x, 1*sx) + *simdStrideAt(y, 1*sy)
			v2 := alpha**simdStrideAt(x, 2*sx) + *simdStrideAt(y, 2*sy)
			v3 := alpha**simdStrideAt(x, 3*sx) + *simdStrideAt(y, 3*sy)
			*simdStrideAt(y, 0*sy) = v0
			*simdStrideAt(y, 1*sy) = v1
			*simdStrideAt(y, 2*sy) = v2
			*simdStrideAt(y, 3*sy) = v3
		}
		x, y = unsafe.Add(x, 4*sx), unsafe.Add(y, 4*sy)
		{
			v0 := alpha**simdStrideAt(x, 0*sx) + *simdStrideAt(y, 0*sy)
			v1 := alpha**simdStrideAt(x, 1*sx) + *simdStrideAt(y, 1*sy)
			v2 := alpha**simdStrideAt(x, 2*sx) + *simdStrideAt(y, 2*sy)
			v3 := alpha**simdStrideAt(x, 3*sx) + *simdStrideAt(y, 3*sy)
			*simdStrideAt(y, 0*sy) = v0
			*simdStrideAt(y, 1*sy) = v1
			*simdStrideAt(y, 2*sy) = v2
			*simdStrideAt(y, 3*sy) = v3
		}
		n -= 8
		if n == 0 {
			return
		}
		x, y = unsafe.Add(x, 4*sx), unsafe.Add(y, 4*sy)
	}
	for n >= 4 {
		v0 := alpha**simdStrideAt(x, 0*sx) + *simdStrideAt(y, 0*sy)
		v1 := alpha**simdStrideAt(x, 1*sx) + *simdStrideAt(y, 1*sy)
		v2 := alpha**simdStrideAt(x, 2*sx) + *simdStrideAt(y, 2*sy)
		v3 := alpha**simdStrideAt(x, 3*sx) + *simdStrideAt(y, 3*sy)
		*simdStrideAt(y, 0*sy) = v0
		*simdStrideAt(y, 1*sy) = v1
		*simdStrideAt(y, 2*sy) = v2
		*simdStrideAt(y, 3*sy) = v3
		n -= 4
		if n == 0 {
			return
		}
		x, y = unsafe.Add(x, 4*sx), unsafe.Add(y, 4*sy)
	}
	for n > 0 {
		*simdStrideAt(y, 0) = alpha**simdStrideAt(x, 0) + *simdStrideAt(y, 0)
		n--
		if n == 0 {
			return
		}
		x, y = unsafe.Add(x, 1*sx), unsafe.Add(y, 1*sy)
	}
}
func axpyIncToPositiveSIMD(alpha float64, x, y, dst unsafe.Pointer, n, sx, sy, sd uintptr) {
	for n >= 4 {
		v0 := alpha**simdStrideAt(x, 0) + *simdStrideAt(y, 0)
		v1 := alpha**simdStrideAt(x, sx) + *simdStrideAt(y, sy)
		v2 := alpha**simdStrideAt(x, 2*sx) + *simdStrideAt(y, 2*sy)
		v3 := alpha**simdStrideAt(x, 3*sx) + *simdStrideAt(y, 3*sy)
		*simdStrideAt(dst, 0) = v0
		*simdStrideAt(dst, sd) = v1
		*simdStrideAt(dst, 2*sd) = v2
		*simdStrideAt(dst, 3*sd) = v3
		n -= 4
		if n == 0 {
			return
		}
		x, y, dst = unsafe.Add(x, 4*sx), unsafe.Add(y, 4*sy), unsafe.Add(dst, 4*sd)
	}
	for n > 0 {
		*simdStrideAt(dst, 0) = alpha**simdStrideAt(x, 0) + *simdStrideAt(y, 0)
		n--
		if n == 0 {
			return
		}
		x, y, dst = unsafe.Add(x, sx), unsafe.Add(y, sy), unsafe.Add(dst, sd)
	}
}
