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

func TestSIMDShortElementwiseGuardPage(t *testing.T) {
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
	xPage, yPage, dstPage := allocate(), allocate(), allocate()
	for n := 0; n <= 65; n++ {
		x, y, dst := xPage[len(xPage)-n:], yPage[len(yPage)-n:], dstPage[len(dstPage)-n:]
		for i := range x {
			x[i], y[i], dst[i] = float64(i+1), 0, -1
		}
		if got := LinfDistSIMD(x, y); got != float64(n) {
			t.Fatalf("Linf n=%d: got %g want %g", n, got, float64(n))
		}
		ScalUnitaryToSIMD(dst, -1, x)
		ScalUnitarySIMD(-1, x)
		for i := range x {
			if x[i] != -float64(i+1) || dst[i] != x[i] {
				t.Fatalf("Scal n=%d index=%d: x=%g dst=%g", n, i, x[i], dst[i])
			}
		}
	}
}
