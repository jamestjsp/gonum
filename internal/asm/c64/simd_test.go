// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"math"
	"simd"
	"slices"
	"testing"
)

func TestPortableComplexLaneBits(t *testing.T) {
	bits := []uint32{0, 0x80000000, 1, 0x3f800000, 0x7f800000, 0xff800000, 0x7fc00042, 0x7f800001}
	width := simd.BroadcastFloat32s(0).Len()
	for start := range bits {
		input, got := make([]uint32, width), make([]uint32, width)
		for i := range input {
			input[i] = bits[(start+i)%len(bits)]
		}
		complexSwapSIMD(simd.LoadUint32s(input).BitsToFloat32()).ToBits().Store(got)
		for i := range input {
			if got[i] != input[i^1] {
				t.Fatalf("start=%d lane=%d: got %x want %x", start, i, got[i], input[i^1])
			}
		}
	}
}

func TestSIMDCandidateComplexLengths(t *testing.T) {
	for n := 0; n <= 129; n++ {
		x, y := make([]complex64, n), make([]complex64, n)
		var wantc, wantu complex64
		for i := range x {
			x[i] = complex(float32(i%13-6)/4, float32(i%7-3)/8)
			y[i] = complex(float32(i%11-5)/8, float32(i%17-8)/4)
			wantc += conj64(x[i]) * y[i]
			wantu += x[i] * y[i]
		}
		if got := DotcUnitarySIMD(x, y); got != wantc {
			t.Fatalf("Dotc n=%d: got %v want %v", n, got, wantc)
		}
		if got := DotuUnitarySIMD(x, y); got != wantu {
			t.Fatalf("Dotu n=%d: got %v want %v", n, got, wantu)
		}
		dst, want := make([]complex64, n+1), make([]complex64, n+1)
		const alpha = complex64(0.75 - 0.25i)
		dst[n], want[n] = 17+19i, 17+19i
		for i := range x {
			want[i] = alpha*x[i] + y[i]
		}
		AxpyUnitaryToSIMD(dst[:n], alpha, x, y)
		if !slices.Equal(dst, want) {
			t.Fatalf("AxpyTo n=%d: got %v want %v", n, dst, want)
		}
		AxpyUnitarySIMD(alpha, x, y)
		if !slices.Equal(y, want[:n]) {
			t.Fatalf("Axpy n=%d: got %v want %v", n, y, want[:n])
		}
	}
}

func TestSIMDCandidateComplexOverlap(t *testing.T) {
	for _, offsets := range [][3]int{{0, 0, 0}, {0, 0, 33}, {33, 0, 33}, {1, 0, 33}, {0, 1, 33}, {1, 33, 0}, {0, 33, 1}} {
		got := make([]complex64, 67)
		for i := range got {
			got[i] = complex(float32(i%13-6)/4, float32(i%7-3)/8)
		}
		want := slices.Clone(got)
		d, x, y := offsets[0], offsets[1], offsets[2]
		const alpha = complex64(0.75 - 0.25i)
		for i := 0; i < 33; i++ {
			want[d+i] = alpha*want[x+i] + want[y+i]
		}
		AxpyUnitaryToSIMD(got[d:d+33], alpha, got[x:x+33], got[y:y+33])
		if !slices.Equal(got, want) {
			t.Fatalf("offsets=%v: got %v want %v", offsets, got, want)
		}
	}
}

func TestSIMDCandidateComplexStrides(t *testing.T) {
	for _, stride := range []uintptr{0, 1, 2, ^uintptr(0)} {
		for _, n := range []uintptr{0, 1, 3, 8, 17} {
			const alpha = complex64(0.75 - 0.25i)
			x, y, dst := make([]complex64, 40), make([]complex64, 40), make([]complex64, 40)
			for i := range x {
				x[i] = complex(float32(i%13-6)/4, float32(i%7-3)/8)
				y[i] = complex(float32(i%11-5)/8, float32(i%17-8)/4)
			}
			start := uintptr(1)
			if stride == ^uintptr(0) {
				start = 20
			}
			if n == 0 {
				start = ^uintptr(0)
			}
			want := slices.Clone(dst)
			var wantc, wantu complex64
			ix := start
			for i := uintptr(0); i < n; i++ {
				want[ix] = alpha*x[ix] + y[ix]
				wantc += conj64(x[ix]) * y[ix]
				wantu += x[ix] * y[ix]
				ix += stride
			}
			AxpyIncToSIMD(dst, stride, start, alpha, x, y, n, stride, stride, start, start)
			if !slices.Equal(dst, want) {
				t.Fatalf("AxpyIncTo n=%d stride=%d", n, stride)
			}
			if got := DotcIncSIMD(x, y, n, stride, stride, start, start); got != wantc {
				t.Fatalf("DotcInc n=%d stride=%d: got %v want %v", n, stride, got, wantc)
			}
			if got := DotuIncSIMD(x, y, n, stride, stride, start, start); got != wantu {
				t.Fatalf("DotuInc n=%d stride=%d: got %v want %v", n, stride, got, wantu)
			}
			want = slices.Clone(y)
			ix = start
			for i := uintptr(0); i < n; i++ {
				want[ix] = alpha*x[ix] + want[ix]
				ix += stride
			}
			AxpyIncSIMD(alpha, x, y, n, stride, stride, start, start)
			if !slices.Equal(y, want) {
				t.Fatalf("AxpyInc n=%d stride=%d", n, stride)
			}
		}
	}
}

func TestSIMDCandidateComplexCancellation(t *testing.T) {
	const n = 4097
	x, y := make([]complex64, n), make([]complex64, n)
	var wantc, wantu complex128
	var scale float64
	for i := range x {
		x[i] = complex(float32(math.Sin(float64(i))), float32(math.Cos(float64(i)*0.3)))
		y[i] = complex(float32(math.Cos(float64(i)*0.2)), float32(math.Sin(float64(i)*0.7)))
		xc, yc := complex128(x[i]), complex128(y[i])
		wantc += complex(real(xc), -imag(xc)) * yc
		wantu += xc * yc
		scale += math.Hypot(real(xc), imag(xc)) * math.Hypot(real(yc), imag(yc))
	}
	for _, test := range []struct {
		got  complex64
		want complex128
	}{{DotcUnitarySIMD(x, y), wantc}, {DotuUnitarySIMD(x, y), wantu}} {
		diff := complex128(test.got) - test.want
		if math.Hypot(real(diff), imag(diff)) > 1e-6*scale {
			t.Fatalf("got %v want %v scale %v", test.got, test.want, scale)
		}
	}
}

func TestSIMDCandidateComplexStrideOverlap(t *testing.T) {
	for _, index := range [][4]uintptr{{1, 2, 0, 0}, {2, 1, 0, 0}, {2, 2, 0, 1}} {
		got := make([]complex64, 40)
		for i := range got {
			got[i] = complex(float32(i%13-6)/4, float32(i%7-3)/8)
		}
		want := slices.Clone(got)
		incX, incY, ix, iy := index[0], index[1], index[2], index[3]
		for i := 0; i < 17; i++ {
			want[iy] = (0.75-0.25i)*want[ix] + want[iy]
			ix += incX
			iy += incY
		}
		AxpyIncSIMD(0.75-0.25i, got, got, 17, incX, incY, index[2], index[3])
		if !slices.Equal(got, want) {
			t.Fatalf("index=%v: got %v want %v", index, got, want)
		}
	}
}

func TestSIMDCandidateComplexMixedStrides(t *testing.T) {
	for _, n := range []uintptr{1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 63} {
		for _, increments := range [][3]int{{3, 7, 5}, {-3, 2, 7}, {0, 7, 3}, {7, -3, 2}} {
			x, y, dst := make([]complex64, 512), make([]complex64, 512), make([]complex64, 512)
			for i := range x {
				x[i] = complex(float32(i%13-6)/4, float32(i%7-3)/8)
				y[i] = complex(float32(i%11-5)/8, float32(i%17-8)/4)
				dst[i] = 17 + 19i
			}
			startX, startY := uintptr(3), uintptr(5)
			if increments[0] < 0 {
				startX += uintptr(-increments[0]) * (n - 1)
			}
			if increments[1] < 0 {
				startY += uintptr(-increments[1]) * (n - 1)
			}
			incX, incY, incDst := uintptr(increments[0]), uintptr(increments[1]), uintptr(increments[2])
			ix, iy, idst := startX, startY, uintptr(2)
			want := slices.Clone(dst)
			var wantc, wantu complex64
			for i := uintptr(0); i < n; i++ {
				want[idst] = (0.75-0.25i)*x[ix] + y[iy]
				wantc += conj64(x[ix]) * y[iy]
				wantu += x[ix] * y[iy]
				ix += incX
				iy += incY
				idst += incDst
			}
			AxpyIncToSIMD(dst, incDst, 2, 0.75-0.25i, x, y, n, incX, incY, startX, startY)
			if !slices.Equal(dst, want) {
				t.Fatalf("Axpy mixed strides n=%d increments=%v", n, increments)
			}
			if got := DotcIncSIMD(x, y, n, incX, incY, startX, startY); got != wantc {
				t.Fatalf("Dotc mixed strides n=%d increments=%v: got %v want %v", n, increments, got, wantc)
			}
			if got := DotuIncSIMD(x, y, n, incX, incY, startX, startY); got != wantu {
				t.Fatalf("Dotu mixed strides n=%d increments=%v: got %v want %v", n, increments, got, wantu)
			}
		}
	}
}
