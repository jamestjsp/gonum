// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && !safe && !noasm && !gccgo

package c128

import (
	"fmt"
	"syscall"
	"testing"
	"unsafe"
)

// Separate protected-mapping supplement, not the ordinary-storage reproducer.
// Alignment 8 has an unavoidable eight-byte accessible gap at a page edge;
// alignment 0 puts the final exact 16-byte element against PROT_NONE.
func axpyAlignmentGuard(t *testing.T, n int, mod uintptr, atEnd bool) []complex128 {
	t.Helper()
	page := syscall.Getpagesize()
	if n < 1 || n*16+8 > page || (mod != 0 && mod != 8) {
		t.Fatal("invalid guarded fixture")
	}
	data, err := syscall.Mmap(-1, 0, 3*page, syscall.PROT_NONE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(data); err != nil {
			t.Error(err)
		}
	})
	if err := syscall.Mprotect(data[page:2*page], syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
		t.Fatal(err)
	}
	for i := page; i < 2*page; i++ {
		data[i] = 0xa5
	}
	start := page + int(mod)
	if atEnd {
		start = 2*page - 16*n - int(mod)
	}
	address := uintptr(unsafe.Pointer(&data[start]))
	if address%16 != mod || address%uintptr(unsafe.Alignof(complex128(0))) != 0 {
		t.Fatal("guard alignment mismatch")
	}
	t.Cleanup(func() {
		for i := page; i < 2*page; i++ {
			if i >= start && i < start+16*n {
				continue
			}
			if data[i] != 0xa5 {
				t.Errorf("guard padding changed at byte %d", i-page)
				break
			}
		}
	})
	return unsafe.Slice((*complex128)(unsafe.Pointer(&data[start])), n)
}

func TestAxpyAlignmentProtectedEnds(t *testing.T) {
	for _, op := range []string{"Unitary", "UnitaryTo", "Inc", "IncTo"} {
		for _, n := range []int{1, 4, 5, 9} {
			for _, mod := range []uintptr{0, 8} {
				for _, reverse := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/n=%d/mod=%d/reverse=%t", op, n, mod, reverse), func(t *testing.T) {
						step := 1
						if op == "Inc" || op == "IncTo" {
							step = 3
						}
						size := (n-1)*step + 1
						starts := [3]int{}
						if reverse && step != 1 {
							step = -step
							starts = [3]int{size - 1, size - 1, size - 1}
						}
						x := axpyAlignmentGuard(t, size, mod, !reverse)
						y := axpyAlignmentGuard(t, size, mod, !reverse)
						dst := axpyAlignmentGuard(t, size, mod, !reverse)
						for i := range x {
							x[i], y[i], dst[i] = complex(float64(i%7-3)/8, 0.25), 0.5-0.125i, -7+2i
						}
						if op == "Unitary" || op == "Inc" {
							dst = y
						}
						wantX, wantY, wantDst := append([]complex128(nil), x...), append([]complex128(nil), y...), append([]complex128(nil), dst...)
						if op == "Unitary" || op == "Inc" {
							wantDst = wantY
						}
						steps := [3]int{step, step, step}
						axpyAlignmentOracle(0.5-0.25i, wantX, wantY, wantDst, n, steps, starts)
						axpyAlignmentCall(op, 0.5-0.25i, x, y, dst, n, steps, starts)
						for i := range x {
							if x[i] != wantX[i] || y[i] != wantY[i] || dst[i] != wantDst[i] {
								t.Fatalf("value/gap mismatch at %d", i)
							}
						}
					})
				}
			}
		}
	}
}
