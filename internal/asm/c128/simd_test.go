// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"math"
	"simd"
	"testing"
)

func TestPortableComplexLaneBits(t *testing.T) {
	bits := []uint64{0, 0x8000000000000000, 1, 0x3ff0000000000000, 0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000042, 0x7ff0000000000001}
	width := simd.BroadcastFloat64s(0).Len()
	for start := range bits {
		src, dst := make([]complex128, width), make([]complex128, width)
		for i := range src {
			src[i] = complex(math.Float64frombits(bits[(start+i)%len(bits)]), math.Float64frombits(bits[(start+i+1)%len(bits)]))
		}
		re, im := make([]uint64, width), make([]uint64, width)
		portableLoadComplex128(src, re, im, width)
		simd.LoadUint64s(re).BitsToFloat64().ToBits().Store(re)
		simd.LoadUint64s(im).BitsToFloat64().ToBits().Store(im)
		portableStoreComplex128(dst, re, im, width)
		for i, value := range dst {
			wantRe, wantIm := bits[(start+i)%len(bits)], bits[(start+i+1)%len(bits)]
			if re[i] != wantRe || im[i] != wantIm || math.Float64bits(real(value)) != wantRe || math.Float64bits(imag(value)) != wantIm {
				t.Errorf("start=%d lane=%d: lane transfer changed component bits", start, i)
			}
		}
	}
}
