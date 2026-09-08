// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"slices"
	"testing"
)

func BenchmarkDgerARM64SIMD(b *testing.B) {
	impl := Implementation{}
	for _, shape := range [][2]int{{3, 7}, {4, 1}, {4, 2}, {4, 3}, {4, 31}, {8, 8}, {8, 31}, {16, 64}, {64, 16}, {64, 64}, {64, 256}, {64, 512}, {128, 32}, {128, 129}} {
		for _, pad := range []int{0, 3} {
			m, n := shape[0], shape[1]
			b.Run(fmt.Sprintf("m=%d/n=%d/pad=%d", m, n, pad), func(b *testing.B) {
				lda := n + pad
				x := make([]float64, m)
				y := make([]float64, n)
				a := make([]float64, (m-1)*lda+n)
				for i := range x {
					x[i] = float64(i%11-5) * 0.125
				}
				for i := range y {
					y[i] = float64(i%7-3) * 0.25
				}
				xOriginal := slices.Clone(x)
				yOriginal := slices.Clone(y)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					alpha := 0.5
					if i&1 != 0 {
						alpha = -alpha
					}
					impl.Dger(m, n, alpha, x, 1, y, 1, a, lda)
				}
				b.StopTimer()
				if !slices.Equal(x, xOriginal) || !slices.Equal(y, yOriginal) {
					b.Fatal("Dger modified an input")
				}
				odd := b.N&1 != 0
				for i := 0; i < m; i++ {
					for j := 0; j < lda && i*lda+j < len(a); j++ {
						want := 0.0
						if odd && j < n {
							want = 0.5 * x[i] * y[j]
						}
						if a[i*lda+j] != want {
							b.Fatalf("A[%d,%d]=%v want %v", i, j, a[i*lda+j], want)
						}
					}
				}
			})
		}
	}
}

func BenchmarkDgerARM64SIMDStridedControl(b *testing.B) {
	const m, n, lda = 16, 64, 67
	impl := Implementation{}
	for _, inc := range [][2]int{{2, 1}, {1, 3}, {2, 3}} {
		incX, incY := inc[0], inc[1]
		b.Run(fmt.Sprintf("incX=%d/incY=%d", incX, incY), func(b *testing.B) {
			x := make([]float64, (m-1)*incX+1)
			y := make([]float64, (n-1)*incY+1)
			a := make([]float64, (m-1)*lda+n)
			for i := 0; i < m; i++ {
				x[i*incX] = float64(i%11-5) * 0.125
			}
			for i := 0; i < n; i++ {
				y[i*incY] = float64(i%7-3) * 0.25
			}
			xOriginal := slices.Clone(x)
			yOriginal := slices.Clone(y)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				alpha := 0.5
				if i&1 != 0 {
					alpha = -alpha
				}
				impl.Dger(m, n, alpha, x, incX, y, incY, a, lda)
			}
			b.StopTimer()
			if !slices.Equal(x, xOriginal) || !slices.Equal(y, yOriginal) {
				b.Fatal("Dger modified an input")
			}
			odd := b.N&1 != 0
			for i := 0; i < m; i++ {
				for j := 0; j < lda && i*lda+j < len(a); j++ {
					want := 0.0
					if odd && j < n {
						want = 0.5 * x[i*incX] * y[j*incY]
					}
					if a[i*lda+j] != want {
						b.Fatalf("A[%d,%d]=%v want %v", i, j, a[i*lda+j], want)
					}
				}
			}
		})
	}
}
