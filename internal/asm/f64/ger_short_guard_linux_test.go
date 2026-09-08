// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestGerEightGuardPage(t *testing.T) {
	page := syscall.Getpagesize()
	allocate := func() []float64 {
		data, err := syscall.Mmap(-1, 0, 2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := syscall.Munmap(data); err != nil {
				t.Error(err)
			}
		})
		if err := syscall.Mprotect(data[page:], syscall.PROT_NONE); err != nil {
			t.Fatal(err)
		}
		return unsafe.Slice((*float64)(unsafe.Pointer(&data[0])), page/8)
	}
	xPage, yPage, aPage := allocate(), allocate(), allocate()
	for m := 1; m <= 32; m++ {
		for _, lda := range []int{8, 11} {
			x, y := xPage[len(xPage)-m:], yPage[len(yPage)-8:]
			n := (m-1)*lda + 8
			a := aPage[len(aPage)-n:]
			for i := range x {
				x[i] = float64(i + 1)
			}
			for i := range y {
				y[i] = float64(i + 1)
			}
			for i := range a {
				a[i] = -99
			}
			GerSIMD(uintptr(m), 8, 0.5, x, 1, y, 1, a, uintptr(lda))
			for i := range a {
				want := -99.0
				if i%lda < 8 {
					want += 0.5 * x[i/lda] * y[i%lda]
				}
				if a[i] != want {
					t.Fatalf("m=%d lda=%d index=%d got=%g want=%g", m, lda, i, a[i], want)
				}
			}
		}
	}
}
