// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"runtime/debug"
	"syscall"
	"testing"
	"unsafe"
)

func TestSIMDSumPackedZeroGuardPages(t *testing.T) {
	previous := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(previous)
	page := syscall.Getpagesize()
	data, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(data); err != nil {
			t.Error(err)
		}
	})
	for _, block := range [][]byte{data[:page], data[2*page:]} {
		if err := syscall.Mprotect(block, syscall.PROT_NONE); err != nil {
			t.Fatal(err)
		}
	}
	values := unsafe.Slice((*float32)(unsafe.Pointer(&data[page])), page/4)
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257} {
		for _, atEnd := range []bool{false, true} {
			x := values[:n:n]
			if atEnd {
				x = values[len(values)-n : len(values) : len(values)]
			}
			var want float32
			for i := range x {
				x[i] = float32(i%7-3) / 8
				want += x[i]
			}
			if got := SumSIMD(x); got != want {
				t.Fatalf("n%d atEnd%t: got%g want%g", n, atEnd, got, want)
			}
		}
	}
}
