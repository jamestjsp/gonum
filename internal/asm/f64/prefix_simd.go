// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import "simd"

// These scalar blocks retain the vector path's addition and multiplication
// order while avoiding staging and loading the intermediate prefix twice.
// Only measured amd64 paths select them; other architectures keep the portable
// vector implementation.
func cumSumBlocksSIMD(dst, src []float64) []float64 {
	width := simd.BroadcastFloat64s(0).Len()
	var sum float64
	var i int
	if width == 8 {
		for ; i+8 <= len(src); i += 8 {
			x, out := src[i:i+8], dst[i:i+8]
			p0 := x[0]
			p1 := x[1] + p0
			p2 := x[2] + p1
			p3 := x[3] + p2
			p4 := x[4] + p3
			p5 := x[5] + p4
			p6 := x[6] + p5
			p7 := x[7] + p6
			out[0] = p0 + sum
			out[1] = p1 + sum
			out[2] = p2 + sum
			out[3] = p3 + sum
			out[4] = p4 + sum
			out[5] = p5 + sum
			out[6] = p6 + sum
			out[7] = p7 + sum
			sum = p7 + sum
		}
	}
	if width == 4 {
		for ; i+4 <= len(src); i += 4 {
			x, out := src[i:i+4], dst[i:i+4]
			p0 := x[0]
			p1 := x[1] + p0
			p2 := x[2] + p1
			p3 := x[3] + p2
			out[0] = p0 + sum
			out[1] = p1 + sum
			out[2] = p2 + sum
			out[3] = p3 + sum
			sum = p3 + sum
		}
	}
	for ; i+width <= len(src); i += width {
		block := src[i]
		dst[i] = block + sum
		for lane := 1; lane < width; lane++ {
			block = src[i+lane] + block
			dst[i+lane] = block + sum
		}
		sum = block + sum
	}
	for ; i < len(src); i++ {
		sum += src[i]
		dst[i] = sum
	}
	return dst
}

func cumProdBlocksSIMD(dst, src []float64) []float64 {
	width := simd.BroadcastFloat64s(0).Len()
	product := 1.0
	var i int
	if width == 8 {
		for ; i+8 <= len(src); i += 8 {
			x, out := src[i:i+8], dst[i:i+8]
			p0 := x[0]
			p1 := x[1] * p0
			p2 := x[2] * p1
			p3 := x[3] * p2
			p4 := x[4] * p3
			p5 := x[5] * p4
			p6 := x[6] * p5
			p7 := x[7] * p6
			out[0] = p0 * product
			out[1] = p1 * product
			out[2] = p2 * product
			out[3] = p3 * product
			out[4] = p4 * product
			out[5] = p5 * product
			out[6] = p6 * product
			out[7] = p7 * product
			product = p7 * product
		}
	}
	if width == 4 {
		for ; i+4 <= len(src); i += 4 {
			x, out := src[i:i+4], dst[i:i+4]
			p0 := x[0]
			p1 := x[1] * p0
			p2 := x[2] * p1
			p3 := x[3] * p2
			out[0] = p0 * product
			out[1] = p1 * product
			out[2] = p2 * product
			out[3] = p3 * product
			product = p3 * product
		}
	}
	for ; i+width <= len(src); i += width {
		block := src[i]
		dst[i] = block * product
		for lane := 1; lane < width; lane++ {
			block = src[i+lane] * block
			dst[i+lane] = block * product
		}
		product = block * product
	}
	for ; i < len(src); i++ {
		product *= src[i]
		dst[i] = product
	}
	return dst
}
