// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"math"
	"runtime/debug"
	"syscall"
	"testing"
	"unsafe"
)

func TestSIMDSumPointerMissingGuards(t *testing.T) {
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
	for _, n := range []int{5, 6, 9, 10, 11, 12, 13, 14} {
		for _, atEnd := range []bool{false, true} {
			t.Run(fmt.Sprintf("n=%d/end=%t", n, atEnd), func(t *testing.T) {
				if err := syscall.Mprotect(data[page:2*page], syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
					t.Fatal(err)
				}
				for i := range values {
					values[i] = math.Float32frombits(0x7fc12345)
				}
				x := values[:n:n]
				if atEnd {
					x = values[len(values)-n : len(values) : len(values)]
				}
				for i := range x {
					x[i] = float32(i%7-3) / 8
				}
				want := sumPortableEntrySIMD(x)
				if err := syscall.Mprotect(data[page:2*page], syscall.PROT_READ); err != nil {
					t.Fatal(err)
				}
				sumPointerEqual(t, "read-only guarded exact-length result", SumSIMD(x), want)
			})
		}
	}
}
