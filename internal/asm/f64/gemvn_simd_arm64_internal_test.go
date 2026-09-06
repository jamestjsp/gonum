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

func TestGemvNStridedOutputSpanOverflow(t *testing.T) {
	const maxUint = ^uintptr(0)
	a, x, y := make([]float64, 32), make([]float64, 8), make([]float64, 7)
	if !gemvNStridedOutputValid(4, 8, a, 8, x, y, 2) {
		t.Fatal("valid boundary rejected")
	}
	for _, tc := range []struct{ m, n, lda, incY uintptr }{
		{4, 8, maxUint, 2},
		{4, 8, 8, maxUint / 2},
		{4, maxUint, maxUint, 2},
		{maxUint, 8, 8, 2},
		{4, 8, 8, maxUint},
	} {
		if gemvNStridedOutputValid(tc.m, tc.n, a, tc.lda, x, y, tc.incY) {
			t.Errorf("invalid spans accepted: %+v", tc)
		}
	}
}

func TestGemvNStridedOutputLargeRows(t *testing.T) {
	for _, m := range []int{33, 64, 255, 256, 257} {
		for _, n := range []int{8, 9, 31, 32, 33, 255} {
			for _, incY := range []int{2, 32} {
				for _, coeff := range [][2]float64{{1, 0}, {-0.75, 1}, {0.5, -0.5}} {
					t.Run(fmt.Sprintf("m=%d/n=%d/incy=%d/alpha=%g/beta=%g", m, n, incY, coeff[0], coeff[1]), func(t *testing.T) {
						const guard = 4
						offset := 1 + (m+n)%3
						lda := n + 5
						aStore := gemvNLargeData(guard + offset + m*lda + guard)
						xStore := gemvNLargeData(guard + offset + n + guard)
						yStore := gemvNLargeData(guard + offset + (m-1)*incY + 1 + guard)
						a := aStore[guard+offset:]
						x := xStore[guard+offset:]
						y := yStore[guard+offset:]
						if coeff[1] == 0 {
							for i := 0; i < m; i++ {
								y[i*incY] = math.NaN()
							}
						}
						aOrig, xOrig := slices.Clone(aStore), slices.Clone(xStore)
						want := slices.Clone(yStore)
						if !gemvNStridedOutputValid(uintptr(m), uintptr(n), a, uintptr(lda), x, y, uintptr(incY)) {
							t.Fatal("large-row geometry rejected")
						}
						gemvN(uintptr(m), uintptr(n), coeff[0], a, uintptr(lda), x, 1, coeff[1], want[guard+offset:], uintptr(incY))
						GemvN(uintptr(m), uintptr(n), coeff[0], a, uintptr(lda), x, 1, coeff[1], y, uintptr(incY))
						gemvNLargeCheckBits(t, yStore, want)
						if !gemvNLargeEqualBits(aStore, aOrig) || !gemvNLargeEqualBits(xStore, xOrig) {
							t.Fatal("read-only input or guard changed")
						}
					})
				}
			}
		}
	}
}

func TestGemvNStridedOutputLargeRowsNumerical(t *testing.T) {
	const m, n, lda, incY = 33, 9, 12, 32
	for _, pattern := range []string{"ordered-overflow", "ordered-cancellation", "subnormal", "signed-zero", "infinity", "nan", "zero-times-infinity"} {
		t.Run(pattern, func(t *testing.T) {
			a := make([]float64, m*lda)
			x := make([]float64, n)
			y := make([]float64, (m-1)*incY+1)
			for i := range x {
				x[i] = 1
			}
			switch pattern {
			case "ordered-overflow":
				a[0], a[1], a[2] = math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
			case "ordered-cancellation":
				a[0], a[1], a[2] = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64
			case "subnormal":
				a[0], a[1], a[2] = 8*math.SmallestNonzeroFloat64, -4*math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64
				x[0], x[1], x[2] = 0.5, 0.5, 1
			case "signed-zero":
				for i := range a {
					a[i] = math.Copysign(0, -1)
				}
			case "infinity":
				a[0] = math.Inf(-1)
			case "nan":
				a[0] = math.NaN()
			case "zero-times-infinity":
				a[0], x[0] = math.Inf(1), 0
			}
			for i := 1; i < m; i++ {
				copy(a[i*lda:i*lda+n], a[:n])
			}
			want := slices.Clone(y)
			gemvN(m, n, 1, a, lda, x, 1, 0, want, incY)
			GemvN(m, n, 1, a, lda, x, 1, 0, y, incY)
			gemvNLargeCheckBits(t, y, want)
		})
	}
}

func TestGemvNStridedOutputLargeRowsAliasFallback(t *testing.T) {
	const m, n, lda, incY = 33, 9, 12, 2
	for _, overlap := range []string{"x", "later-a-row"} {
		t.Run(overlap, func(t *testing.T) {
			var shared, a, x, y []float64
			var aoff, xoff, yoff int
			if overlap == "x" {
				shared = gemvNLargeData(128)
				a = gemvNLargeData(m * lda)
				xoff, yoff = 0, 1
				x, y = shared[xoff:], shared[yoff:]
			} else {
				shared = gemvNLargeData(m*lda + 80)
				aoff, yoff = 0, 16*lda
				a, x, y = shared[aoff:], gemvNLargeData(n), shared[yoff:]
			}
			if gemvNStridedOutputValid(m, n, a, lda, x, y, incY) {
				t.Fatal("accepted active output alias")
			}
			want := slices.Clone(shared)
			wa, wx, wy := a, x, want[yoff:]
			if overlap == "x" {
				wx = want[xoff:]
			} else {
				wa = want[aoff:]
			}
			gemvN(m, n, -0.75, wa, lda, wx, 1, -0.5, wy, incY)
			GemvN(m, n, -0.75, a, lda, x, 1, -0.5, y, incY)
			gemvNLargeCheckBits(t, shared, want)
		})
	}
}

func TestGemvNStridedOutputLargeRowsIncrementFallback(t *testing.T) {
	const m, n, lda = 33, 9, 12
	for _, tc := range []struct{ incX, incY int }{{0, 32}, {-1, 32}, {2, 32}, {1, 0}, {1, -2}} {
		t.Run(fmt.Sprintf("incx=%d/incy=%d", tc.incX, tc.incY), func(t *testing.T) {
			a := gemvNLargeData(m * lda)
			x := dlarftRowWiseVector(n, tc.incX)
			y := dlarftRowWiseVector(m, tc.incY)
			want := slices.Clone(y)
			gemvN(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, want, uintptr(tc.incY))
			GemvN(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, y, uintptr(tc.incY))
			gemvNLargeCheckBits(t, y, want)
		})
	}
}

func TestGemvNStridedOutputLargeRowsShortSlices(t *testing.T) {
	const m, n, lda, incY = 33, 9, 12, 2
	aNeed, yNeed := (m-1)*lda+n, (m-1)*incY+1
	for _, tc := range []struct {
		name             string
		aLen, xLen, yLen int
	}{
		{"short-a", aNeed - 1, n, yNeed},
		{"short-x", aNeed, n - 1, yNeed},
		{"short-y", aNeed, n, yNeed - 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gaStore, gxStore, gyStore := gemvNLargeData(aNeed+4), gemvNLargeData(n+4), gemvNLargeData(yNeed+4)
			waStore, wxStore, wyStore := slices.Clone(gaStore), slices.Clone(gxStore), slices.Clone(gyStore)
			ga, gx, gy := gaStore[:tc.aLen], gxStore[:tc.xLen], gyStore[:tc.yLen]
			wa, wx, wy := waStore[:tc.aLen], wxStore[:tc.xLen], wyStore[:tc.yLen]
			if gemvNStridedOutputValid(m, n, ga, lda, gx, gy, incY) {
				t.Fatal("accepted short active slice")
			}
			wantPanic := gemvNLargePanics(func() { gemvN(m, n, -0.75, wa, lda, wx, 1, -0.5, wy, incY) })
			gotPanic := gemvNLargePanics(func() { GemvN(m, n, -0.75, ga, lda, gx, 1, -0.5, gy, incY) })
			if gotPanic != wantPanic {
				t.Fatalf("panic=%t want %t", gotPanic, wantPanic)
			}
			gemvNLargeCheckBits(t, gaStore, waStore)
			gemvNLargeCheckBits(t, gxStore, wxStore)
			gemvNLargeCheckBits(t, gyStore, wyStore)
		})
	}
}

func gemvNLargePanics(fn func()) (panicked bool) {
	defer func() {
		panicked = recover() != nil
	}()
	fn()
	return false
}

func gemvNLargeData(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%29-14) / 16
	}
	return x
}

func gemvNLargeCheckBits(t *testing.T, got, want []float64) {
	t.Helper()
	if !gemvNLargeEqualBits(got, want) {
		for i := range got {
			if math.Float64bits(got[i]) != math.Float64bits(want[i]) && !(math.IsNaN(got[i]) && math.IsNaN(want[i])) {
				t.Fatalf("[%d]: got %g (%#x) want %g (%#x)", i, got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
			}
		}
	}
}

func gemvNLargeEqualBits(x, y []float64) bool {
	return slices.EqualFunc(x, y, func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b) || math.IsNaN(a) && math.IsNaN(b)
	})
}
