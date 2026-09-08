// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64 && !safe && !noasm && !gccgo

package gonum

import (
	"runtime/debug"
	"syscall"
	"testing"
	"unsafe"
)

func TestGerNegativeGuarded(t *testing.T) {
	t.Run("Sger", func(t *testing.T) { testGerNegativeStrides(t, Implementation{}.Sger, guardedGerNegative[float32]) })
	t.Run("Dger", func(t *testing.T) { testGerNegativeStrides(t, Implementation{}.Dger, guardedGerNegative[float64]) })
}
func guardedGerNegative[T ~float32 | ~float64](t *testing.T, count int) []T {
	t.Helper()
	restore := debug.SetPanicOnFault(true)
	t.Cleanup(func() { debug.SetPanicOnFault(restore) })
	size := int(unsafe.Sizeof(T(0)))
	page := syscall.Getpagesize()
	used := (count*size + page - 1) / page * page
	memory, err := syscall.Mmap(-1, 0, used+page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(memory); err != nil {
			t.Error(err)
		}
	})
	if err := syscall.Mprotect(memory[used:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&memory[used-count*size])), count)
}
