// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"syscall"
	"testing"
	"unsafe"
)

func TestL1MultiGuardPages(t *testing.T) {
	page := syscall.Getpagesize()
	for _, n := range []int{1, 7, 8, 9, 31, 32, 33, 127, 128, 129, 143, 144, 145, 159, 160, 161, 255, 256, 257, 4095, 4096, 4097, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158} {
		for _, side := range []string{"front", "back"} {
			t.Run(fmt.Sprintf("n%d/%s", n, side), func(t *testing.T) {
				accessible := (8*n + page - 1) / page * page
				data, err := syscall.Mmap(-1, 0, accessible+2*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := syscall.Munmap(data); err != nil {
						t.Error(err)
					}
				}()
				if err := syscall.Mprotect(data[:page], syscall.PROT_NONE); err != nil {
					t.Fatal(err)
				}
				if err := syscall.Mprotect(data[page+accessible:], syscall.PROT_NONE); err != nil {
					t.Fatal(err)
				}
				start := page
				if side == "back" {
					start = page + accessible - 8*n
				}
				// Both mappings satisfy float64 alignment. Back cases end
				// exactly at the guard; front cases start exactly after it.
				x := unsafe.Slice((*float64)(unsafe.Pointer(&data[start])), n)
				for i := range x {
					x[i] = -0.25
				}
				if err := syscall.Mprotect(data[page:page+accessible], syscall.PROT_READ); err != nil {
					t.Fatal(err)
				}
				want := float64(n) / 4
				l1MultiSame(t, L1NormSIMD(x), want)
				l1MultiSame(t, L1NormIncSIMD(x, n, 1), want)
			})
		}
	}
}
