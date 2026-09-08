// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
)

func gerTailMergeCheck(t *testing.T, m, n, incX, incY, mode int) {
	t.Helper()
	span := func(n, inc int) (int, int) {
		if inc < 0 {
			return 1 - (n-1)*inc, -(n - 1) * inc
		}
		return 1 + (n-1)*inc, 0
	}
	xLen, ix := span(m, incX)
	yLen, iy := span(n, incY)
	lda := n + 3
	x, y := make([]float32, xLen), make([]float32, yLen)
	storage := make([]float32, (m-1)*lda+n+4)
	a := storage[2 : len(storage)-2 : len(storage)-2]
	for i := range storage {
		storage[i] = -99
	}
	for i := range x {
		x[i] = math.Float32frombits(0x7fc00042)
	}
	for i := range y {
		y[i] = math.Float32frombits(0x7fc00084)
	}
	alpha := float32(1)
	for i := 0; i < m; i++ {
		x[ix+i*incX] = 1
	}
	for j := 0; j < n; j++ {
		y[iy+j*incY] = float32(j%11-5) * .125
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			value := float32((i+j)%13-6) * .0625
			switch mode {
			case 1: // A duplicated update would change finite to +Inf.
				value = .25 * math.MaxFloat32
				y[iy+j*incY] = .5 * math.MaxFloat32
			case 2:
				value = -.75 * math.MaxFloat32
				y[iy+j*incY] = .75 * math.MaxFloat32
			case 3:
				alpha = math.Float32frombits(1 << 31)
				value = alpha
				y[iy+j*incY] = 1
			case 4: // Only A supplies a quiet NaN; its payload must survive.
				value = math.Float32frombits(0x7fc00100 + uint32(j))
				y[iy+j*incY] = 1
			case 5:
				value = float32(math.Inf(-1))
				y[iy+j*incY] = float32(math.Inf(1))
			case 6: // alpha=0 must not bypass 0*Inf arithmetic.
				alpha = 0
				y[iy+j*incY] = float32(math.Inf(1))
			}
			a[i*lda+j] = value
		}
	}
	want := slices.Clone(storage)
	xBefore, yBefore := slices.Clone(x), slices.Clone(y)
	for i := 0; i < m; i++ {
		scale := float32(alpha * x[ix+i*incX])
		for j := 0; j < n; j++ {
			product := float32(y[iy+j*incY] * scale)
			index := 2 + i*lda + j
			want[index] = float32(product + want[index])
		}
	}
	GerSIMD(uintptr(m), uintptr(n), alpha, x, uintptr(incX), y, uintptr(incY), a, uintptr(lda))
	for i := range want {
		gotBits, wantBits := math.Float32bits(storage[i]), math.Float32bits(want[i])
		if gotBits == wantBits {
			continue
		}
		if mode >= 5 && math.IsNaN(float64(storage[i])) && math.IsNaN(float64(want[i])) {
			continue
		}
		t.Fatalf("storage[%d]: got%08x want%08x", i, gotBits, wantBits)
	}
	for i := range x {
		if math.Float32bits(x[i]) != math.Float32bits(xBefore[i]) {
			t.Fatalf("x[%d] changed", i)
		}
	}
	for i := range y {
		if math.Float32bits(y[i]) != math.Float32bits(yBefore[i]) {
			t.Fatalf("y[%d] changed", i)
		}
	}
}

func TestSIMDGerTailMergeExact(t *testing.T) {
	for _, m := range []int{4, 8, 64, 65} {
		for _, n := range []int{13, 14, 15, 21, 22, 23, 29, 30, 31, 61, 62, 63, 69, 70, 71, 125, 126, 127, 32, 64, 65, 129} {
			for _, inc := range [][2]int{{1, 1}, {2, 3}, {3, 7}, {7, 2}} {
				for mode := 0; mode < 7; mode++ {
					t.Run(fmt.Sprintf("m=%d/n=%d/inc=%d,%d/mode=%d", m, n, inc[0], inc[1], mode), func(t *testing.T) { gerTailMergeCheck(t, m, n, inc[0], inc[1], mode) })
				}
			}
		}
	}
	for _, n := range []int{21, 31, 63} {
		for _, inc := range [][2]int{{-2, 3}, {2, -3}, {0, 0}} {
			for mode := 0; mode < 7; mode++ {
				t.Run(fmt.Sprintf("fallback/n=%d/inc=%d,%d/mode=%d", n, inc[0], inc[1], mode), func(t *testing.T) { gerTailMergeCheck(t, 8, n, inc[0], inc[1], mode) })
			}
		}
	}
}

func TestSIMDGerTailMergeIndependentGaps(t *testing.T) {
	for _, n := range []int{21, 22, 23, 29, 30, 31, 61, 62, 63, 125, 126, 127} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			const m, inc = 8, 7
			lda := n + 3
			x, y, a := make([]float32, (m-1)*inc+1), make([]float32, (n-1)*inc+1), make([]float32, (m-1)*lda+n)
			for i := 0; i < m; i++ {
				x[i*inc] = 1
			}
			for j := 0; j < n; j++ {
				y[j*inc] = 2
			}
			done := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					select {
					case <-done:
						return
					default:
					}
					for i := range x {
						if i%inc != 0 {
							x[i]++
						}
					}
					for i := range y {
						if i%inc != 0 {
							y[i]++
						}
					}
					for i := range a {
						if i%lda >= n {
							a[i]++
						}
					}
				}
			}()
			for range 50 {
				GerSIMD(m, uintptr(n), .5, x, inc, y, inc, a, uintptr(lda))
			}
			close(done)
			wg.Wait()
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					if a[i*lda+j] != 50 {
						t.Fatalf("A[%d,%d]=%g", i, j, a[i*lda+j])
					}
				}
			}
		})
	}
}
