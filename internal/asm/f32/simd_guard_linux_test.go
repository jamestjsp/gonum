// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"runtime/debug"
	"syscall"
	"testing"
	"unsafe"
)

func guardedSIMDF32(t *testing.T, n int) []float32 {
	t.Helper()
	size := syscall.Getpagesize()
	if n*4 > size {
		t.Fatal("guard fixture too large")
	}
	memory, err := syscall.Mmap(-1, 0, 2*size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(memory); err != nil {
			t.Error(err)
		}
	})
	if err := syscall.Mprotect(memory[size:], syscall.PROT_NONE); err != nil {
		t.Fatal(err)
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&memory[size-4*n])), n)
}

// Go 1.27.1 LoadFloat32sPart can lower to an unmasked full-vector load
// followed by register masking. Exact-capacity Go slices do not reveal it;
// placing the final element against an inaccessible page does.
func TestSIMDGuardedTails(t *testing.T) {
	restore := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(restore)
	for _, n := range []int{1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33} {
		x, y, dst := guardedSIMDF32(t, n), guardedSIMDF32(t, n), guardedSIMDF32(t, n)
		for i := range x {
			x[i], y[i] = 1, 2
		}
		AxpyUnitaryToSIMD(dst, 0.5, x, y)
		AxpyUnitarySIMD(0.5, x, y)
		for i := range x {
			if y[i] != 2.5 || dst[i] != 2.5 {
				t.Fatalf("AXPY n=%d i=%d", n, i)
			}
		}
		if got := DotUnitarySIMD(x, y); got != float32(n)*2.5 {
			t.Fatalf("Dot n=%d got%g", n, got)
		}
		if got := SumSIMD(x); got != float32(n) {
			t.Fatalf("Sum n=%d got%g", n, got)
		}
		const m = 4
		a, gx, gy := guardedSIMDF32(t, m*n), guardedSIMDF32(t, m), guardedSIMDF32(t, n)
		for i := range gx {
			gx[i] = 1
		}
		for i := range gy {
			gy[i] = 2
		}
		GerSIMD(m, uintptr(n), 0.5, gx, 1, gy, 1, a, uintptr(n))
		for i, v := range a {
			if v != 1 {
				t.Fatalf("Ger n=%d i=%d got%g", n, i, v)
			}
		}
	}
}
