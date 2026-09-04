// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd"
	"testing"
)

func TestPortableComplexLaneBits(t *testing.T) {
	bits := []uint32{0, 0x80000000, 1, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc00042, 0x7f800001}
	width := simd.BroadcastFloat32s(0).Len()
	for start := range bits {
		src, dst := make([]complex64, width), make([]complex64, width)
		for i := range src {
			src[i] = complex(math.Float32frombits(bits[(start+i)%len(bits)]), math.Float32frombits(bits[(start+i+1)%len(bits)]))
		}
		re, im := make([]uint32, width), make([]uint32, width)
		portableLoadComplex64(src, re, im, width)
		simd.LoadUint32s(re).BitsToFloat32().ToBits().Store(re)
		simd.LoadUint32s(im).BitsToFloat32().ToBits().Store(im)
		portableStoreComplex64(dst, re, im, width)
		for i, value := range dst {
			wantRe, wantIm := bits[(start+i)%len(bits)], bits[(start+i+1)%len(bits)]
			if re[i] != wantRe || im[i] != wantIm || math.Float32bits(real(value)) != wantRe || math.Float32bits(imag(value)) != wantIm {
				t.Errorf("start=%d lane=%d: lane transfer changed component bits", start, i)
			}
		}
	}
}
