// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"syscall"
	"testing"
	"unsafe"
)

func TestGerTiledProtectedGaps(t *testing.T) {
	page := syscall.Getpagesize()
	allocate := func(count int, sparse bool) ([]float64, int) {
		pages := 2
		if sparse {
			pages = 2 * count
		}
		data, err := syscall.Mmap(-1, 0, pages*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := syscall.Munmap(data); err != nil {
				t.Error(err)
			}
		})
		for i := 1; i < pages; i += 2 {
			if err := syscall.Mprotect(data[i*page:(i+1)*page], syscall.PROT_NONE); err != nil {
				t.Fatal(err)
			}
		}
		if sparse {
			return unsafe.Slice((*float64)(unsafe.Pointer(&data[page-8])), (count-1)*2*page/8+1), 2 * page / 8
		}
		return unsafe.Slice((*float64)(unsafe.Pointer(&data[page-count*8])), count), 1
	}
	for _, n := range []int{4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33} {
		for _, sparseY := range []bool{true, false} {
			const m = 4
			x, sx := allocate(m, true)
			y, sy := allocate(n, sparseY)
			lda := n + 2
			a, _ := allocate((m-1)*lda+n, false)
			for i := 0; i < m; i++ {
				x[i*sx] = float64(i + 1)
			}
			for j := 0; j < n; j++ {
				y[j*sy] = float64(j + 1)
			}
			for i := range a {
				a[i] = -99
			}
			GerSIMD(m, uintptr(n), 0.5, x, uintptr(sx), y, uintptr(sy), a, uintptr(lda))
			for i, v := range a {
				want := -99.0
				if i%lda < n {
					want += 0.5 * float64(i/lda+1) * float64(i%lda+1)
				}
				if v != want {
					t.Fatalf("n=%d sparseY=%t index=%d got=%g want=%g", n, sparseY, i, v, want)
				}
			}
		}
	}
}
