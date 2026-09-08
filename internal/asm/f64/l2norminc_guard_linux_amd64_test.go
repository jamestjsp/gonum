// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && linux && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"syscall"
	"testing"
	"unsafe"
)

func TestL2NormIncHardwareGuardPage(t *testing.T) {
	page := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	if err := syscall.Mprotect(data[page:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	for n := 16; n <= 65; n++ {
		for _, inc := range []int{2, 3, 7} {
			length := (n-1)*inc + 1
			x := unsafe.Slice((*float64)(unsafe.Pointer(&data[page-8*length])), length)
			for i := range x {
				x[i] = math.NaN()
			}
			for i := 0; i < n; i++ {
				x[i*inc] = 1
			}
			if got, ok := l2NormIncHardwareSIMD(x, uintptr(n), uintptr(inc)); ok {
				checkNativeNormULP(t, got, math.Sqrt(float64(n)))
			}
		}
	}
}

func TestNormShortHardwareGuardPage(t *testing.T) {
	page := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	if err := syscall.Mprotect(data[page:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	for n := 1; n <= 129; n++ {
		x := unsafe.Slice((*float64)(unsafe.Pointer(&data[page-8*n])), n)
		y := make([]float64, n)
		for i := range x {
			x[i] = 1
		}
		checkNativeNormULP(t, L2NormUnitarySIMD(x), math.Sqrt(float64(n)))
		checkNativeNormULP(t, L2DistanceUnitarySIMD(x, y), math.Sqrt(float64(n)))
		checkNativeNormULP(t, L2DistanceUnitarySIMD(y, x), math.Sqrt(float64(n)))
		if got := L1NormSIMD(x); got != float64(n) {
			t.Fatalf("L1Norm n=%d got=%g", n, got)
		}
		if got := L1DistSIMD(y, x); got != float64(n) {
			t.Fatalf("L1Dist n=%d got=%g", n, got)
		}
		if got := L1DistSIMD(x, y); got != float64(n) {
			t.Fatalf("L1Dist reversed n=%d got=%g", n, got)
		}
		for i := range y {
			y[i] = 2
		}
		DivToSIMD(y, y, x)
		DivToSIMD(x, y, y)
		DivSIMD(x, y)
		for i, v := range x {
			if v != 0.5 || y[i] != 2 {
				t.Fatalf("Div n=%d index=%d x=%g y=%g", n, i, v, y[i])
			}
		}
	}
}
