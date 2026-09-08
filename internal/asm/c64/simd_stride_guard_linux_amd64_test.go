// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"reflect"
	"testing"
)

func TestSIMDComplexStrideGuardPages(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65} {
		for _, reverse := range []bool{false, true} {
			size := 1
			if n > 0 {
				size += 3 * (n - 1)
			}
			x := complexGuardedSliceSIMD(t, size, true)
			y := complexGuardedSliceSIMD(t, size, false)
			dst := complexGuardedSliceSIMD(t, size, true)
			want := make([]complex64, size)
			for i := range x {
				x[i], y[i] = complex(float32(i%7+1), -0.5), 2+0.25i
				dst[i], want[i] = -123, -123
			}
			inc, start := uintptr(3), uintptr(0)
			if reverse {
				inc, start = ^uintptr(2), uintptr(size-1)
			}
			var dotu, dotc complex64
			index := start
			for i := 0; i < n; i++ {
				want[index] = (0.25-0.5i)*x[index] + y[index]
				dotu += x[index] * y[index]
				dotc += conj64(x[index]) * y[index]
				index += inc
			}
			AxpyIncToSIMD(dst, inc, start, 0.25-0.5i, x, y, uintptr(n), inc, inc, start, start)
			if !reflect.DeepEqual(dst, want) {
				t.Fatalf("AXPY n=%d reverse=%t changed values or gaps", n, reverse)
			}
			if got := DotuIncSIMD(x, y, uintptr(n), inc, inc, start, start); got != dotu {
				t.Fatalf("Dotu n=%d reverse=%t: got %v want %v", n, reverse, got, dotu)
			}
			if got := DotcIncSIMD(x, y, uintptr(n), inc, inc, start, start); got != dotc {
				t.Fatalf("Dotc n=%d reverse=%t: got %v want %v", n, reverse, got, dotc)
			}
			// Exercise the public in-place route independently of AxpyIncTo:
			// it may use a different native leaf with y as its write stream.
			wantY := append([]complex64(nil), y...)
			for j, index := 0, start; j < n; j, index = j+1, index+inc {
				wantY[index] = (0.25-0.5i)*x[index] + wantY[index]
			}
			AxpyIncSIMD(0.25-0.5i, x, y, uintptr(n), inc, inc, start, start)
			if !reflect.DeepEqual(y, wantY) {
				t.Fatalf("in-place AXPY n=%d reverse=%t changed values or gaps", n, reverse)
			}
		}
	}
}
