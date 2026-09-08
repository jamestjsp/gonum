// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c128

import (
	"reflect"
	"syscall"
	"testing"
	"unsafe"
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
			want := make([]complex128, size)
			for i := range x {
				x[i], y[i] = complex(float64(i%7+1), -0.5), 2+0.25i
				dst[i], want[i] = -123, -123
			}
			inc, start := uintptr(3), uintptr(0)
			if reverse {
				inc, start = ^uintptr(2), uintptr(size-1)
			}
			var dotu, dotc complex128
			index := start
			for i := 0; i < n; i++ {
				want[index] = (0.25-0.5i)*x[index] + y[index]
				dotu += x[index] * y[index]
				dotc += conj128(x[index]) * y[index]
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
			if reverse {
				continue // Scaling starts at index zero and accepts a positive increment.
			}
			want = append(want[:0], x...)
			for i := 0; i < n; i++ {
				want[3*i] *= 0.25 - 0.5i
			}
			ScalIncSIMD(0.25-0.5i, x, uintptr(n), 3)
			if !reflect.DeepEqual(x, want) {
				t.Fatalf("Scal n=%d changed values or gaps", n)
			}
			for i := 0; i < n; i++ {
				want[3*i] *= 0.5
			}
			DscalIncSIMD(0.5, x, uintptr(n), 3)
			if !reflect.DeepEqual(x, want) {
				t.Fatalf("Dscal n=%d changed values or gaps", n)
			}
		}
	}
}

func complexGuardedSliceSIMD(t *testing.T, n int, atEnd bool) []complex128 {
	t.Helper()
	page := syscall.Getpagesize()
	if n*16 > page {
		t.Fatal("guard fixture exceeds a page")
	}
	data, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_NONE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(data); err != nil {
			t.Error(err)
		}
	})
	if err := syscall.Mprotect(data[page:2*page], syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
		t.Fatal(err)
	}
	start := page
	if atEnd {
		start = 2*page - 16*n
	}
	return unsafe.Slice((*complex128)(unsafe.Pointer(&data[start])), n)
}
