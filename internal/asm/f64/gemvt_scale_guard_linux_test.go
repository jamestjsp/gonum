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

func TestGemvTScaleGuardPage(t *testing.T) {
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
	xp, ap, yp := allocate(), allocate(), allocate()
	for _, m := range []int{1, 3, 4, 5} {
		for _, n := range []int{1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129} {
			lda := n + 3
			span := (m-1)*lda + n
			if span > len(ap) {
				continue
			}
			x, a, y := xp[len(xp)-m:], ap[len(ap)-span:], yp[len(yp)-n:]
			for i := range x {
				x[i] = float64(i%3-1) * .25
			}
			for i := range a {
				a[i] = float64(i%11-5) * .0625
			}
			for i := range y {
				y[i] = float64(i%7-3) * .125
			}
			want := append([]float64(nil), y...)
			for j := range want {
				want[j] *= .25
			}
			for i := 0; i < m; i++ {
				scale := float64(.5 * x[i])
				for j := 0; j < n; j++ {
					product := float64(scale * a[i*lda+j])
					want[j] += product
				}
			}
			GemvTSIMD(uintptr(m), uintptr(n), .5, a, uintptr(lda), x, 1, .25, y, 1)
			checkGemvTEightBits(t, y, want)
		}
	}
}
