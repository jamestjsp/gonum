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

func TestGerARM64SIMDExceptional(t *testing.T) {
	for n := 2; n <= 9; n++ {
		for _, m := range []int{4, 5} {
			for _, pad := range []int{0, 3} {
				lda := n + pad
				for _, pattern := range []string{"fma", "nonfinite", "signed-zero", "alpha-zero", "alpha-zero-infinity"} {
					t.Run(fmt.Sprintf("m=%d/n=%d/pad=%d/%s", m, n, pad, pattern), func(t *testing.T) {
						x, y, a, alpha := gerARM64ExceptionalData(m, n, lda, pattern)
						xBefore, yBefore := slices.Clone(x), slices.Clone(y)
						want := slices.Clone(a)
						gerARM64Reference(m, n, alpha, slices.Clone(x), slices.Clone(y), want, lda)

						got := slices.Clone(a)
						if !gerARM64SIMD(uintptr(m), uintptr(n), alpha, x, 1, y, 1, got, uintptr(lda)) {
							t.Fatal("supported operands rejected")
						}
						checkGerARM64Slice(t, "direct", got, want)
						checkGerARM64Slice(t, "direct x", x, xBefore)
						checkGerARM64Slice(t, "direct y", y, yBefore)

						got = slices.Clone(a)
						Ger(uintptr(m), uintptr(n), alpha, x, 1, y, 1, got, uintptr(lda))
						checkGerARM64Slice(t, "public", got, want)
						checkGerARM64Slice(t, "public x", x, xBefore)
						checkGerARM64Slice(t, "public y", y, yBefore)
					})
				}
			}
		}
	}
}

func TestGerARM64SIMDAliasFallback(t *testing.T) {
	const m = 5
	for n := 2; n <= 9; n++ {
		for _, pad := range []int{0, 3} {
			lda := n + pad
			aLen := (m-1)*lda + n
			for _, alias := range []string{"x", "y"} {
				t.Run(fmt.Sprintf("n=%d/pad=%d/%s", n, pad, alias), func(t *testing.T) {
					got := make([]float64, aLen+8)
					for i := range got {
						got[i] = float64(i%11-5) * 0.125
					}
					got[1] = 1 + 0x1p-27
					got[2] = 1 - 0x1p-27
					want := slices.Clone(got)
					x := []float64{1, -2, math.Inf(1), 0, math.Copysign(0, -1)}
					y := make([]float64, n)
					for i := range y {
						y[i] = float64(i%5-2) * 0.25
					}
					var gx, gy, wx, wy []float64
					if alias == "x" {
						gx, wx = got[1:1+m], want[1:1+m]
						gy, wy = y, slices.Clone(y)
					} else {
						gx, wx = x, slices.Clone(x)
						gy, wy = got[1:1+n], want[1:1+n]
					}
					if gerARM64SIMD(m, uintptr(n), -0.75, gx, 1, gy, 1, got[:aLen], uintptr(lda)) {
						t.Fatal("accepted aliased operands")
					}
					gerARM64Reference(m, n, -0.75, wx, wy, want[:aLen], lda)
					Ger(m, uintptr(n), -0.75, gx, 1, gy, 1, got[:aLen], uintptr(lda))
					checkGerARM64Slice(t, "alias fallback", got, want)
				})
			}
		}
	}
}

func gerARM64Reference(m, n int, alpha float64, x, y, a []float64, lda int) {
	for i, xv := range x[:m] {
		AxpyUnitary(alpha*xv, y[:n], a[i*lda:i*lda+n])
	}
}

func gerARM64ExceptionalData(m, n, lda int, pattern string) (x, y, a []float64, alpha float64) {
	x = make([]float64, m)
	y = make([]float64, n)
	a = make([]float64, (m-1)*lda+n)
	for i := range x {
		x[i] = float64(i%5-2) * 0.25
	}
	for i := range y {
		y[i] = float64(i%7-3) * 0.125
	}
	for i := range a {
		a[i] = float64(i%11-5) * 0.0625
	}
	alpha = -0.75
	switch pattern {
	case "fma":
		alpha = 1 + 0x1p-27
		x[0], y[0], a[0] = 1-0x1p-27, 1+0x1p-27, -1
	case "nonfinite":
		x[0], x[1] = math.Inf(1), math.NaN()
		y[0], y[n-1] = 0, math.Inf(-1)
		a[0], a[n-1] = math.Inf(1), math.NaN()
	case "signed-zero":
		alpha = math.Copysign(0, -1)
		for i := range x {
			x[i] = math.Copysign(0, -1)
		}
		for i := range y {
			y[i] = math.Copysign(0, -1)
		}
		for i := range a {
			a[i] = math.Copysign(0, -1)
		}
	case "alpha-zero":
		alpha = 0
	case "alpha-zero-infinity":
		alpha = 0
		x[0], y[n-1] = math.Inf(1), math.Inf(-1)
	}
	return x, y, a, alpha
}

func checkGerARM64Slice(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length: got %d want %d", name, len(got), len(want))
	}
	for i, g := range got {
		w := want[i]
		if math.Float64bits(g) != math.Float64bits(w) && !(math.IsNaN(g) && math.IsNaN(w)) {
			t.Fatalf("%s index=%d: got=%g (%x) want=%g (%x)", name, i, g, math.Float64bits(g), w, math.Float64bits(w))
		}
	}
}
