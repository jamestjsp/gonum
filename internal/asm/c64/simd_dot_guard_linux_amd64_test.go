// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestSIMDComplexShortDotGuardPages(t *testing.T) {
	// Exercise every remainder at both short-kernel cutoffs. The inputs touch
	// inaccessible pages on opposite ends so a vector tail cannot overread.
	for n := 0; n <= 65; n++ {
		for _, xAtEnd := range []bool{false, true} {
			x := complexGuardedSliceSIMD(t, n, xAtEnd)
			y := complexGuardedSliceSIMD(t, n, !xAtEnd)
			var wantc, wantu complex64
			for i := range x {
				x[i] = complex(float32(i%13-6)/4, float32(i%7-3)/8)
				y[i] = complex(float32(i%11-5)/8, float32(i%17-8)/4)
				wantc += conj64(x[i]) * y[i]
				wantu += x[i] * y[i]
			}
			if got := DotcUnitarySIMD(x, y); got != wantc {
				t.Fatalf("Dotc n=%d xAtEnd=%t: got %v want %v", n, xAtEnd, got, wantc)
			}
			if got := DotuUnitarySIMD(x, y); got != wantu {
				t.Fatalf("Dotu n=%d xAtEnd=%t: got %v want %v", n, xAtEnd, got, wantu)
			}
			dst := complexGuardedSliceSIMD(t, n, xAtEnd)
			want := make([]complex64, n)
			for i := range want {
				want[i] = (0.75-0.25i)*x[i] + y[i]
			}
			AxpyUnitaryToSIMD(dst, 0.75-0.25i, x, y)
			AxpyUnitarySIMD(0.75-0.25i, x, y)
			for i, v := range want {
				if dst[i] != v || y[i] != v {
					t.Fatalf("Axpy n=%d xAtEnd=%t i=%d: got (%v,%v) want %v", n, xAtEnd, i, dst[i], y[i], v)
				}
			}

		}
	}
}

func complexGuardedSliceSIMD(t *testing.T, n int, atEnd bool) []complex64 {
	t.Helper()
	page := syscall.Getpagesize()
	if n*8 > page {
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
		start = 2*page - 8*n
	}
	return unsafe.Slice((*complex64)(unsafe.Pointer(&data[start])), n)
}
