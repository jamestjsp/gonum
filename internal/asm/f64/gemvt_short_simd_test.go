// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"slices"
	"testing"
)

// Independent scalar oracle, including the effect of aliases on later inputs.
func gemvTEightReference(m, lda int, alpha float64, a, x []float64, beta float64, y []float64) {
	for j := 0; j < 8; j++ {
		if beta == 0 {
			y[j] = 0
		} else {
			y[j] = float64(beta * y[j])
		}
	}
	for i := 0; i < m; i++ {
		scale := float64(alpha * x[i])
		for j := 0; j < 8; j++ {
			product := float64(scale * a[i*lda+j])
			y[j] += product
		}
	}
}

func checkGemvTEightBits(t *testing.T, got, want []float64) {
	t.Helper()
	for i, g := range got {
		w := want[i]
		if math.Float64bits(g) != math.Float64bits(w) && !(math.IsNaN(g) && math.IsNaN(w)) {
			t.Fatalf("index=%d got=%g (%x) want=%g (%x)", i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}

func TestGemvTEightIEEEAndPadding(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), 1, -3, math.SmallestNonzeroFloat64, math.MaxFloat64, math.Inf(1), math.Inf(-1), math.NaN()}
	for _, m := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 64} {
		for _, lda := range []int{8, 11} {
			for _, alpha := range values {
				for _, beta := range []float64{0, math.Copysign(0, -1), .25, -1, math.Inf(1), math.NaN()} {
					x, a, y := make([]float64, m), make([]float64, max(0, (m-1)*lda+8)), make([]float64, 11)
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
					gemvTEightReference(m, lda, alpha, a, x, beta, want)
					GemvTSIMD(uintptr(m), 8, alpha, a, uintptr(lda), x, 1, beta, y, 1)
					checkGemvTEightBits(t, y, want)
					checkGemvTEightBits(t, a, aBefore)
					checkGemvTEightBits(t, x, xBefore)
				}
			}
		}
	}
}

func TestGemvTEightRowOrder(t *testing.T) {
	for _, m := range []int{3, 4, 5, 8, 9} {
		const lda = 11
		x, a, y := make([]float64, m), make([]float64, (m-1)*lda+8), make([]float64, 8)
		for i := range x {
			x[i] = 1
			for j := 0; j < 8; j++ {
				a[i*lda+j] = []float64{.75 * math.MaxFloat64, -.75 * math.MaxFloat64, .75 * math.MaxFloat64}[i%3]
			}
		}
		want := slices.Clone(y)
		gemvTEightReference(m, lda, 1, a, x, 0, want)
		GemvTSIMD(uintptr(m), 8, 1, a, lda, x, 1, 0, y, 1)
		checkGemvTEightBits(t, y, want)
	}
}

func TestGemvTEightAliasedInputs(t *testing.T) {
	for _, m := range []int{1, 3, 4, 5, 8} {
		for _, lda := range []int{8, 11} {
			for _, alias := range []string{"x", "a", "both"} {
				for _, offset := range []int{0, 1, 7} {
					data := make([]float64, max((m-1)*lda+8, m, offset+8)+8)
					for i := range data {
						data[i] = float64(i%13-6) * .125
					}
					wantData := slices.Clone(data)
					a, x := make([]float64, (m-1)*lda+8), make([]float64, m)
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
					y, wy := data[offset:offset+8], wantData[offset:offset+8]
					gemvTEightReference(m, lda, .5, wa, wx, -.25, wy)
					GemvTSIMD(uintptr(m), 8, .5, a, uintptr(lda), x, 1, -.25, y, 1)
					checkGemvTEightBits(t, data, wantData)
				}
			}
		}
	}
}
