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

func TestGemvTSmallGuardPage(t *testing.T) {
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
	xPage, aPage, yPage := allocate(), allocate(), allocate()
	for n := 4; n <= 16; n++ {
		for m := 1; m <= 17; m++ {
			for _, lda := range []int{n, n + 3} {
				for _, beta := range []float64{0, -.25} {
					x, a, y := xPage[len(xPage)-m:], aPage[len(aPage)-((m-1)*lda+n):], yPage[len(yPage)-n:]
					for i := range x {
						x[i] = float64(i%7-3) * .125
					}
					for i := range a {
						a[i] = float64(i%13-6) * .0625
					}
					for i := range y {
						y[i] = float64(i%5-2) * .25
						if beta == 0 {
							y[i] = math.NaN()
						}
					}
					want := append([]float64(nil), y...)
					gemvTSmallReference(m, n, lda, .5, a, x, beta, want)
					GemvTSIMD(uintptr(m), uintptr(n), .5, a, uintptr(lda), x, 1, beta, y, 1)
					checkGemvTEightBits(t, y, want)
				}
			}
		}
	}
}
