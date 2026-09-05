// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestGemvSIMDDispatch(t *testing.T) {
	for _, shape := range [][2]int{
		{0, 0}, {0, 9}, {1, 0}, {1, 1}, {4, 8},
		{7, 8}, {8, 7}, {8, 8}, {8, 9}, {9, 8},
		{8, 31}, {8, 32}, {8, 33}, {8, 511}, {8, 512}, {8, 513}, {9, 512},
		{15, 16}, {15, 17}, {15, 255}, {15, 256}, {16, 255}, {16, 256},
		{31, 16}, {31, 17}, {32, 16}, {32, 17}, {31, 255}, {32, 255},
		{256, 16}, {64, 64},
	} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, trans := range []bool{false, true} {
			xn, yn, fn, ref := n, m, GemvN, gemvN
			if trans {
				xn, yn, fn, ref = m, n, GemvT, gemvT
			}
			for _, incs := range [][2]int{{1, 1}, {2, 1}, {1, 2}, {-1, 1}, {1, -2}, {0, 1}, {1, 0}} {
				incX, incY := incs[0], incs[1]
				for _, beta := range []float64{0, -0.5, 1} {
					t.Run(fmt.Sprintf("%dx%d/T=%t/x=%d/y=%d/beta=%g", m, n, trans, incX, incY, beta), func(t *testing.T) {
						x, _ := matrixVector(xn, incX)
						y, _ := matrixVector(yn, incY)
						if beta == 0 {
							for i := range y {
								y[i] = math.NaN()
							}
						}
						y = append(y, 7, 8, 9)
						a, _ := matrixVector(m*lda, 1)
						want := slices.Clone(y)
						ref(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, want, uintptr(incY))
						fn(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
						matrixCheck(t, y, want)
					})
				}
			}
		}
	}
}

func TestGemvSIMDDispatchOverlap(t *testing.T) {
	const m, n, lda = 8, 8, 10
	for _, trans := range []bool{false, true} {
		fn, ref := GemvN, gemvN
		if trans {
			fn, ref = GemvT, gemvT
		}
		for _, offset := range []int{0, 1, 17} {
			for _, matrixOverlap := range []bool{false, true} {
				for _, beta := range []float64{0, 0.5} {
					t.Run(fmt.Sprintf("T=%t/offset=%d/matrix=%t/beta=%g", trans, offset, matrixOverlap, beta), func(t *testing.T) {
						got, _ := matrixVector(m*lda+4, 1)
						want := slices.Clone(got)
						a, _ := matrixVector(m*lda, 1)
						wa := a
						if matrixOverlap {
							a, wa = got[2:], want[2:]
						}
						ref(m, n, 0.75, wa, lda, want[2:2+n], 1, beta, want[offset:offset+n], 1)
						fn(m, n, 0.75, a, lda, got[2:2+n], 1, beta, got[offset:offset+n], 1)
						portableSIMDCheckBits(t, "backing slice", got, want)
					})
				}
			}
		}
	}
}

func BenchmarkGemvSIMDDispatch(b *testing.B) {
	for _, shape := range [][2]int{
		{1, 64}, {4, 8}, {7, 8}, {8, 7}, {8, 8}, {8, 9}, {9, 8},
		{8, 16}, {16, 8}, {7, 9}, {64, 64}, {8, 128}, {8, 256},
		{15, 16}, {15, 17}, {15, 255}, {15, 256}, {16, 255}, {16, 256},
		{31, 16}, {31, 17}, {32, 16}, {32, 17}, {31, 255}, {32, 255},
		{256, 16}, {8, 512}, {8, 1024}, {9, 512}, {16, 512}, {16, 1024},
		{32, 512}, {512, 8}, {512, 512},
	} {
		m, n := shape[0], shape[1]
		for _, trans := range []bool{false, true} {
			for _, implementation := range []string{"fallback", "dispatch", "candidate"} {
				b.Run(fmt.Sprintf("%dx%d/T=%t/implementation=%s", m, n, trans, implementation), func(b *testing.B) {
					lda := n + 3
					a, _ := matrixVector(m*lda, 1)
					x, _ := matrixVector(max(m, n), 1)
					y, _ := matrixVector(max(m, n), 1)
					fn := gemvN
					if trans {
						fn = gemvT
					}
					if implementation == "dispatch" {
						fn = GemvN
						if trans {
							fn = GemvT
						}
					} else if implementation == "candidate" {
						fn = GemvNSIMD
						if trans {
							fn = GemvTSIMD
						}
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						fn(uintptr(m), uintptr(n), 0.75, a, uintptr(lda), x, 1, -1, y, 1)
					}
				})
			}
		}
	}
}
