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

func gemvTSmallReference(m, n, lda int, alpha float64, a, x []float64, beta float64, y []float64) {
	for j := 0; j < n; j++ {
		if beta == 0 {
			y[j] = 0
		} else {
			y[j] = float64(beta * y[j])
		}
	}
	for i := 0; i < m; i++ {
		scale := float64(alpha * x[i])
		for j := 0; j < n; j++ {
			product := float64(scale * a[i*lda+j])
			y[j] += product
		}
	}
}

func checkGemvTSmallBits(t *testing.T, context string, got, want []float64) {
	t.Helper()
	for i, g := range got {
		w := want[i]
		if math.Float64bits(g) != math.Float64bits(w) && !(math.IsNaN(g) && math.IsNaN(w)) {
			t.Fatalf("%s: index=%d got=%g (%x) want=%g (%x)", context, i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}

func TestGemvTSmallIEEEAndPadding(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -3, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	for n := 4; n <= 16; n++ {
		for _, m := range []int{0, 1, 2, 3, 4, 5, 8, 9, 17} {
			for _, pad := range []int{0, 3} {
				lda := n + pad
				for _, alpha := range values {
					for _, beta := range []float64{0, math.Copysign(0, -1), .25, -1, math.Inf(1), math.NaN()} {
						x, a, y := make([]float64, m), make([]float64, max(0, (m-1)*lda+n)), make([]float64, n+3)
						for i := range x {
							x[i] = values[i%len(values)]
						}
						for i := range a {
							a[i] = values[(i+3)%len(values)]
						}
						for i := range y {
							y[i] = values[(i+6)%len(values)]
						}
						xBefore, aBefore, want := slices.Clone(x), slices.Clone(a), slices.Clone(y)
						gemvTSmallBackendReference(m, n, lda, alpha, a, x, beta, want)
						GemvTSIMD(uintptr(m), uintptr(n), alpha, a, uintptr(lda), x, 1, beta, y, 1)
						context := fmt.Sprintf("m=%d n=%d pad=%d alpha=%g beta=%g", m, n, pad, alpha, beta)
						checkGemvTSmallBits(t, context, y, want)
						checkGemvTEightBits(t, a, aBefore)
						checkGemvTEightBits(t, x, xBefore)
					}
				}
			}
		}
	}
}

func TestGemvTSmallRowOrder(t *testing.T) {
	for n := 4; n <= 16; n++ {
		for _, m := range []int{3, 4, 5, 8, 9, 17} {
			lda := n + 3
			x, a, y := make([]float64, m), make([]float64, (m-1)*lda+n), make([]float64, n)
			for i := range x {
				x[i] = 1
				for j := 0; j < n; j++ {
					a[i*lda+j] = []float64{.75 * math.MaxFloat64, -.75 * math.MaxFloat64, .75 * math.MaxFloat64}[i%3]
				}
			}
			want := slices.Clone(y)
			gemvTSmallBackendReference(m, n, lda, 1, a, x, 0, want)
			GemvTSIMD(uintptr(m), uintptr(n), 1, a, uintptr(lda), x, 1, 0, y, 1)
			checkGemvTSmallBits(t, fmt.Sprintf("m=%d n=%d row order", m, n), y, want)
		}
	}
}

func TestGemvTSmallAliasedInputs(t *testing.T) {
	for n := 4; n <= 16; n++ {
		for _, m := range []int{1, 3, 4, 5, 8} {
			for _, pad := range []int{0, 3} {
				lda := n + pad
				for _, alias := range []string{"x", "a", "both"} {
					for _, offset := range []int{0, 1, n - 1} {
						data := make([]float64, max((m-1)*lda+n, m, offset+n)+n)
						for i := range data {
							data[i] = float64(i%13-6) * .125
						}
						wantData := slices.Clone(data)
						a, x := make([]float64, (m-1)*lda+n), make([]float64, m)
						for i := range a {
							a[i] = float64(i%7-3) * .0625
						}
						for i := range x {
							x[i] = float64(i%5-2) * .25
						}
						wa, wx := slices.Clone(a), slices.Clone(x)
						if alias == "a" || alias == "both" {
							a, wa = data[:len(a)], wantData[:len(a)]
						}
						if alias == "x" || alias == "both" {
							x, wx = data[:m], wantData[:m]
						}
						y, wy := data[offset:offset+n], wantData[offset:offset+n]
						gemvTSmallReference(m, n, lda, .5, wa, wx, -.25, wy)
						GemvTSIMD(uintptr(m), uintptr(n), .5, a, uintptr(lda), x, 1, -.25, y, 1)
						checkGemvTEightBits(t, data, wantData)
					}
				}
			}
		}
	}
}
