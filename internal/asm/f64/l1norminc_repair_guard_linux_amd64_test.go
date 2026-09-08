// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && !noasm && !safe && !gccgo

package f64

import (
	"fmt"
	"math"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

func TestL1NormIncRepairGuardPages(t *testing.T) {
	lengths := []int{1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65}
	for _, n := range lengths {
		for _, inc := range []int{1, 2, 3, 7} {
			for _, edge := range []string{"lower", "upper"} {
				t.Run(fmt.Sprintf("n=%d/inc=%d/%s", n, inc, edge), func(t *testing.T) {
					page := os.Getpagesize()
					span := 1 + (n-1)*inc
					if span*8 > page {
						t.Fatal("fixed guard shape exceeds one page")
					}
					memory, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_NONE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := syscall.Munmap(memory); err != nil {
							t.Errorf("munmap: %v", err)
						}
					})
					middle := memory[page : 2*page]
					if err := syscall.Mprotect(middle, syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
						t.Fatal(err)
					}
					storage := unsafe.Slice((*float64)(unsafe.Pointer(&middle[0])), page/8)
					for i := range storage {
						storage[i] = math.Float64frombits(0x7ff8000000008000 + uint64(i))
					}
					start := 0
					if edge == "upper" {
						start = len(storage) - span
					}
					x := storage[start : start+span : start+span]
					for k := 0; k < n; k++ {
						x[k*inc] = float64(1+k%31) / 64
						if k%2 != 0 {
							x[k*inc] = -x[k*inc]
						}
					}
					before := l1IncRepairBits(storage)
					want := l1IncRepairScalar(x, n, inc)
					got := L1NormInc(x, n, inc)
					l1IncRepairCheck(t, storage, before, got, want)
				})
			}
		}
	}
}
