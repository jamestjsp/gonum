// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"slices"
	"testing"
)

// matrixVector includes increment gaps, so every comparison also verifies that
// untouched entries and matrix padding are preserved.
func matrixVector(n, inc int) ([]float64, int) {
	size, first := 1, 0
	if n > 0 {
		size += (n - 1) * matrixAbs(inc)
	}
	if inc < 0 {
		first = size - 1
	}
	v := make([]float64, size)
	for i := range v {
		v[i] = float64(i%19-9) * 0.125
	}
	return v, first
}

func matrixCheck(t *testing.T, got, want []float64) {
	t.Helper()
	for i, v := range want {
		g := got[i]
		if g == v || math.IsNaN(float64(g)) && math.IsNaN(float64(v)) {
			continue
		}
		if math.Abs(float64(g-v)) <= 1e-12*(1+math.Abs(float64(v))) {
			continue
		}
		t.Fatalf("element %d: got %.18g want %.18g", i, g, v)
	}
}

func TestSIMDMatrixGer(t *testing.T) {
	for _, shape := range [][2]int{{0, 0}, {0, 9}, {1, 0}, {1, 1}, {2, 7}, {3, 9}, {4, 16}, {5, 17}, {7, 31}, {8, 32}, {9, 33}, {16, 65}, {65, 7}, {64, 64}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, incX := range []int{-2, -1, 0, 1, 2} {
			for _, incY := range []int{-2, -1, 0, 1, 2} {
				t.Run(fmt.Sprintf("%dx%d/x=%d/y=%d", m, n, incX, incY), func(t *testing.T) {
					x, ix := matrixVector(m, incX)
					y, iy := matrixVector(n, incY)
					a, _ := matrixVector(m*lda, 1)
					want := slices.Clone(a)
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							want[i*lda+j] += (-0.75 * x[ix+i*incX]) * y[iy+j*incY]
						}
					}
					GerSIMD(uintptr(m), uintptr(n), -0.75, x, uintptr(incX), y, uintptr(incY), a, uintptr(lda))
					matrixCheck(t, a, want)
				})
			}
		}
	}
}

func TestSIMDMatrixGerOverlap(t *testing.T) {
	for _, offset := range []int{0, 1, 7} {
		const m, n, lda = 8, 17, 19
		a, _ := matrixVector(m*lda, 1)
		want := slices.Clone(a)
		x, _ := matrixVector(m, 1)
		y := a[offset : offset+n]
		wy := want[offset : offset+n]
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				want[i*lda+j] += (-0.75 * x[i]) * wy[j]
			}
		}
		GerSIMD(m, n, -0.75, x, 1, y, 1, a, lda)
		matrixCheck(t, a, want)
	}
}

func TestSIMDMatrixGemv(t *testing.T) {
	for _, shape := range [][2]int{{0, 0}, {0, 9}, {1, 0}, {1, 1}, {2, 7}, {3, 9}, {4, 16}, {5, 17}, {7, 31}, {8, 32}, {9, 33}, {16, 65}, {65, 7}, {64, 64}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, trans := range []bool{false, true} {
			xn, yn := n, m
			if trans {
				xn, yn = m, n
			}
			for _, incX := range []int{-2, -1, 0, 1, 2} {
				for _, incY := range []int{-2, -1, 0, 1, 2} {
					for _, beta := range []float64{0, -0.5, 1} {
						t.Run(fmt.Sprintf("%dx%d/T=%t/x=%d/y=%d/beta=%g", m, n, trans, incX, incY, beta), func(t *testing.T) {
							x, ix := matrixVector(xn, incX)
							y, iy := matrixVector(yn, incY)
							if beta == 0 {
								for i := range y {
									y[i] = math.NaN()
								}
							}
							a, _ := matrixVector(m*lda, 1)
							want := slices.Clone(y)
							if !trans {
								for i := 0; i < m; i++ {
									var sum float64
									for j := 0; j < n; j++ {
										sum += a[i*lda+j] * x[ix+j*incX]
									}
									if beta == 0 {
										want[iy+i*incY] = -0.75 * sum
									} else {
										want[iy+i*incY] = beta*want[iy+i*incY] - 0.75*sum
									}
								}
								GemvNSIMD(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
							} else {
								for j := 0; j < n; j++ {
									if beta == 0 {
										want[iy+j*incY] = 0
									} else {
										want[iy+j*incY] *= beta
									}
								}
								for i := 0; i < m; i++ {
									for j := 0; j < n; j++ {
										want[iy+j*incY] += (-0.75 * x[ix+i*incX]) * a[i*lda+j]
									}
								}
								GemvTSIMD(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
							}
							matrixCheck(t, y, want)
						})
					}
				}
			}
		}
	}
}

func TestSIMDMatrixGemvOverlap(t *testing.T) {
	const m, n, lda = 8, 8, 8
	for _, trans := range []bool{false, true} {
		for _, offset := range []int{0, 1, 7} {
			a, _ := matrixVector(m*lda, 1)
			want := slices.Clone(a)
			x := a[:n]
			y := a[offset : offset+n]
			wx := want[:n]
			wy := want[offset : offset+n]
			if !trans {
				for i := 0; i < m; i++ {
					var dot float64
					for j := 0; j < n; j++ {
						dot += wx[j] * want[i*lda+j]
					}
					wy[i] = 0.5*wy[i] + 0.75*dot
				}
				GemvNSIMD(m, n, 0.75, a, lda, x, 1, 0.5, y, 1)
			} else {
				for j := range wy {
					wy[j] *= 0.5
				}
				for i := 0; i < m; i++ {
					scale := 0.75 * wx[i]
					for j := 0; j < n; j++ {
						wy[j] += scale * want[i*lda+j]
					}
				}
				GemvTSIMD(m, n, 0.75, a, lda, x, 1, 0.5, y, 1)
			}
			matrixCheck(t, a, want)
		}
	}
}

func BenchmarkSIMDMatrixShapes(b *testing.B) {
	for _, shape := range [][2]int{{7, 9}, {8, 8}, {64, 64}, {16, 256}, {256, 16}, {512, 512}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, kernel := range []string{"Ger", "GemvN", "GemvT"} {
			for _, impl := range []string{"current", "simd"} {
				b.Run(fmt.Sprintf("%s/%dx%d/implementation=%s", kernel, m, n, impl), func(b *testing.B) {
					a, _ := matrixVector(m*lda, 1)
					x, _ := matrixVector(max(m, n), 1)
					y, _ := matrixVector(max(m, n), 1)
					var run func()
					switch kernel {
					case "Ger":
						fn := Ger
						if impl == "simd" {
							fn = GerSIMD
						}
						run = func() { fn(uintptr(m), uintptr(n), 0.75, x, 1, y, 1, a, uintptr(lda)) }
					case "GemvN":
						fn := GemvN
						if impl == "simd" {
							fn = GemvNSIMD
						}
						run = func() { fn(uintptr(m), uintptr(n), 0.75, a, uintptr(lda), x, 1, -1, y, 1) }
					case "GemvT":
						fn := GemvT
						if impl == "simd" {
							fn = GemvTSIMD
						}
						run = func() { fn(uintptr(m), uintptr(n), 0.75, a, uintptr(lda), x, 1, -1, y, 1) }
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						run()
					}
				})
			}
		}
	}
}

func matrixAbs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
