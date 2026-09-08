// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"syscall"
	"testing"
	"unsafe"
)

func TestSIMDPositiveStrideGuardPage(t *testing.T) {
	page := syscall.Getpagesize()
	guarded := func(length int) []float64 {
		data, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { syscall.Munmap(data) })
		if err := syscall.Mprotect(data[2*page:], syscall.PROT_NONE); err != nil {
			t.Fatal(err)
		}
		return unsafe.Slice((*float64)(unsafe.Pointer(&data[2*page-8*length])), length)
	}
	for _, n := range []int{1, 2, 3, 4, 6, 7, 8, 9, 18, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129} {
		for _, inc := range []int{1, 2, 3, 7} {
			length := (n-1)*inc + 1
			x, y, dst := guarded(length), guarded(length), guarded(length)
			reset := func() {
				for i := range x {
					x[i], y[i], dst[i] = math.NaN(), math.NaN(), math.NaN()
				}
				for i := 0; i < n; i++ {
					x[i*inc], y[i*inc], dst[i*inc] = 2, 3, 5
				}
			}
			check := func(name string, a []float64, want float64) {
				for i, v := range a {
					if i%inc == 0 {
						if v != want {
							t.Fatalf("%s n=%d inc=%d index=%d got=%g want=%g", name, n, inc, i, v, want)
						}
					} else if !math.IsNaN(v) {
						t.Fatalf("%s wrote gap at %d", name, i)
					}
				}
			}
			reset()
			if inc == 1 {
				if got := DotUnitarySIMD(x, y); got != float64(6*n) {
					t.Fatalf("Dot n=%d got=%g", n, got)
				}
				if got := SumSIMD(x); got != float64(2*n) {
					t.Fatalf("Sum n=%d got=%g", n, got)
				}
			}
			if got := L1NormIncSIMD(x, n, inc); got != float64(2*n) {
				t.Fatalf("L1 n=%d inc=%d got=%g", n, inc, got)
			}
			AxpyIncSIMD(0.5, x, y, uintptr(n), uintptr(inc), uintptr(inc), 0, 0)
			check("Axpy", y, 4)
			reset()
			AxpyIncToSIMD(dst, uintptr(inc), 0, 0.5, x, y, uintptr(n), uintptr(inc), uintptr(inc), 0, 0)
			check("AxpyTo", dst, 4)
			reset()
			ScalIncSIMD(0.5, x, uintptr(n), uintptr(inc))
			check("Scal", x, 1)
			reset()
			ScalIncToSIMD(dst, uintptr(inc), 0.5, x, uintptr(n), uintptr(inc))
			check("ScalTo", dst, 1)
		}
	}
}
