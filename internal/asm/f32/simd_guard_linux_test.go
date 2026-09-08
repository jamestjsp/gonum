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
	used := (n*4 + size - 1) / size * size
	memory, err := syscall.Mmap(-1, 0, used+size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
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
	return unsafe.Slice((*float32)(unsafe.Pointer(&memory[used-4*n])), n)
}

// Go 1.27.1 LoadFloat32sPart can lower to an unmasked full-vector load
// followed by register masking. Exact-capacity Go slices do not reveal it;
// placing the final element against an inaccessible page does.
func TestSIMDGuardedTails(t *testing.T) {
	restore := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(restore)
	for n := 1; n <= 65; n++ {
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
		if got := DdotUnitarySIMD(x, y); got != float64(n)*2.5 {
			t.Fatalf("Ddot n=%d got%g", n, got)
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

// Exercise the validated strided path with its last addressed scalar adjacent
// to an inaccessible page. Every gap remains independently owned storage.
func TestSIMDGuardedPositiveStrides(t *testing.T) {
	restore := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(restore)
	for _, n := range []int{15, 16, 17, 31, 33, 127, 128, 129} {
		for _, inc := range []int{2, 3, 7} {
			span := (n-1)*inc + 1
			x, y, dst := guardedSIMDF32(t, span), guardedSIMDF32(t, span), guardedSIMDF32(t, span)
			for i := range x {
				x[i], y[i], dst[i] = -99, -99, -99
			}
			for i := 0; i < n; i++ {
				x[i*inc], y[i*inc] = 1, 2
			}
			if got := DotIncSIMD(x, y, uintptr(n), uintptr(inc), uintptr(inc), 0, 0); got != float32(2*n) {
				t.Fatalf("Dot n=%d inc=%d got%v", n, inc, got)
			}
			if got := DdotIncSIMD(x, y, uintptr(n), uintptr(inc), uintptr(inc), 0, 0); got != float64(2*n) {
				t.Fatalf("Ddot n=%d inc=%d got%v", n, inc, got)
			}
			AxpyIncToSIMD(dst, uintptr(inc), 0, 0.5, x, y, uintptr(n), uintptr(inc), uintptr(inc), 0, 0)
			AxpyIncSIMD(0.5, x, y, uintptr(n), uintptr(inc), uintptr(inc), 0, 0)
			for i := range x {
				want := float32(-99)
				if i%inc == 0 {
					want = 2.5
				}
				if y[i] != want || dst[i] != want {
					t.Fatalf("AXPY n=%d inc=%d index%d y%v dst%v want%v", n, inc, i, y[i], dst[i], want)
				}
			}
		}
	}
}

func TestSIMDGuardedStridedGer(t *testing.T) {
	restore := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(restore)
	for _, shape := range [][2]int{{1, 4}, {3, 7}, {4, 8}, {5, 9}, {4, 31}, {5, 65}, {4, 129}, {8, 64}, {8, 65}, {8, 129}, {65, 31}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, inc := range []int{3, 7} {
			x := guardedSIMDF32(t, (m-1)*inc+1)
			y := guardedSIMDF32(t, (n-1)*inc+1)
			a := guardedSIMDF32(t, (m-1)*lda+n)
			for i := range x {
				x[i] = -99
			}
			for i := range y {
				y[i] = -99
			}
			for i := range a {
				a[i] = -99
			}
			for i := 0; i < m; i++ {
				x[i*inc] = 1
				for j := 0; j < n; j++ {
					a[i*lda+j] = 0
				}
			}
			for j := 0; j < n; j++ {
				y[j*inc] = 2
			}
			GerSIMD(uintptr(m), uintptr(n), 0.5, x, uintptr(inc), y, uintptr(inc), a, uintptr(lda))
			for i, value := range a {
				want := float32(-99)
				if i%lda < n {
					want = 1
				}
				if value != want {
					t.Fatalf("Ger %dx%d inc%d index%d: got%v want%v", m, n, inc, i, value, want)
				}
			}
		}
	}
}
