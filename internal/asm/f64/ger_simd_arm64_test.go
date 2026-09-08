// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"slices"
	"testing"
)

func TestGerARM64SIMD(t *testing.T) {
	for _, shape := range [][2]int{{4, 2}, {4, 3}, {5, 7}, {8, 16}, {9, 31}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		x := make([]float64, m)
		y := make([]float64, n)
		a := make([]float64, (m-1)*lda+n)
		for i := range x {
			x[i] = float64(i-2) * 0.25
		}
		for i := range y {
			y[i] = float64(i%5-2) * 0.125
		}
		for i := range a {
			a[i] = float64(i%7-3) * 0.0625
		}
		want := slices.Clone(a)
		for i, xv := range x {
			AxpyUnitary(-0.75*xv, y, want[i*lda:i*lda+n])
		}
		Ger(uintptr(m), uintptr(n), -0.75, x, 1, y, 1, a, uintptr(lda))
		for i, got := range a {
			if math.Float64bits(got) != math.Float64bits(want[i]) {
				t.Fatalf("%dx%d element %d: got %v want %v", m, n, i, got, want[i])
			}
		}
	}
}

func TestGerARM64SIMDRejectsUnsupported(t *testing.T) {
	const m, n, lda = 4, 7, 9
	a := make([]float64, (m-1)*lda+n)
	x := make([]float64, m)
	y := make([]float64, n)
	for _, test := range []struct {
		name string
		x    []float64
		y    []float64
		a    []float64
		incX uintptr
		incY uintptr
		lda  uintptr
	}{
		{name: "A aliases y", x: x, y: a[1 : 1+n], a: a, incX: 1, incY: 1, lda: lda},
		{name: "A aliases x", x: a[1 : 1+m], y: y, a: a, incX: 1, incY: 1, lda: lda},
		{name: "strided x", x: x, y: y, a: a, incX: 2, incY: 1, lda: lda},
		{name: "strided y", x: x, y: y, a: a, incX: 1, incY: 2, lda: lda},
		{name: "short x", x: x[:m-1], y: y, a: a, incX: 1, incY: 1, lda: lda},
		{name: "short y", x: x, y: y[:n-1], a: a, incX: 1, incY: 1, lda: lda},
		{name: "short A", x: x, y: y, a: a[:len(a)-1], incX: 1, incY: 1, lda: lda},
		{name: "lda smaller than n", x: x, y: y, a: a, incX: 1, incY: 1, lda: n - 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			if gerARM64SIMD(m, n, 0.5, test.x, test.incX, test.y, test.incY, test.a, test.lda) {
				t.Fatal("accepted unsupported operands")
			}
		})
	}
	if gerARM64SIMD(m, n, 0.5, x, 1, y, 1, a, ^uintptr(0)) {
		t.Fatal("accepted overflowing matrix span")
	}
}
