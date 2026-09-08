// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package c64

import (
	"fmt"
	"syscall"
	"testing"
	"unsafe"
)

func TestC64UnitaryCachedGuardPages(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 7, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 510, 511, 512, 513, 4096} {
		for _, atEnd := range []bool{false, true} {
			t.Run(fmt.Sprintf("n=%d/end=%t", n, atEnd), func(t *testing.T) {
				alloc := func(end bool) ([]complex64, []byte) {
					page := syscall.Getpagesize()
					pages := (n*8 + page - 1) / page
					if pages == 0 {
						pages = 1
					}
					data, err := syscall.Mmap(-1, 0, (pages+2)*page, syscall.PROT_NONE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := syscall.Munmap(data); err != nil {
							t.Error(err)
						}
					})
					active := data[page : (pages+1)*page]
					if err := syscall.Mprotect(active, syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
						t.Fatal(err)
					}
					start := page
					if end {
						start = (pages+1)*page - 8*n
					}
					return unsafe.Slice((*complex64)(unsafe.Pointer(&data[start])), n), active
				}
				x, xpages := alloc(atEnd)
				y, ypages := alloc(!atEnd)
				var rc, ic, ru, iu int64
				for i := range x {
					xr, xi, yr, yi := int64(i%17-8), int64((3*i+1)%19-9), int64((5*i+2)%23-11), int64((7*i+3)%29-14)
					x[i] = complex(float32(xr)/16, float32(xi)/16)
					y[i] = complex(float32(yr)/32, float32(yi)/32)
					rc += xr*yr + xi*yi
					ic += xr*yi - xi*yr
					ru += xr*yr - xi*yi
					iu += xr*yi + xi*yr
				}
				// Both accessible regions become read-only before either call.
				for _, p := range [][]byte{xpages, ypages} {
					if err := syscall.Mprotect(p, syscall.PROT_READ); err != nil {
						t.Fatal(err)
					}
				}
				if got, want := DotcUnitarySIMD(x, y), complex(float32(rc)/512, float32(ic)/512); got != want {
					t.Fatalf("Dotc got=%v want=%v", got, want)
				}
				if got, want := DotuUnitarySIMD(x, y), complex(float32(ru)/512, float32(iu)/512); got != want {
					t.Fatalf("Dotu got=%v want=%v", got, want)
				}
			})
		}
	}
}
