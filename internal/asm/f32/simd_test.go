// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"math"
	"simd"
	"slices"
	"testing"
)

func TestPortableSIMDReductions(t *testing.T) {
	width := simd.BroadcastFloat32s(0).Len()
	for n := 0; n <= 9*width+1; n++ {
		x := make([]float32, n+5)
		y := make([]float32, n+7)
		for i := range x {
			x[i] = float32(i%7-3) / 4
		}
		for i := range y {
			y[i] = float32(i%5-2) / 8
		}
		var dot, sum float32
		var ddot float64
		for i := 0; i < n; i++ {
			dot += x[i+2] * y[i+3]
			ddot += float64(x[i+2]) * float64(y[i+3])
			sum += x[i+2]
		}
		if got := DotUnitarySIMD(x[2:2+n], y[3:]); got != dot {
			t.Errorf("DotUnitarySIMD n=%d: got %v want %v", n, got, dot)
		}
		if got := DotIncSIMD(x, y, uintptr(n), 1, 1, 2, 3); got != dot {
			t.Errorf("DotIncSIMD n=%d: got %v want %v", n, got, dot)
		}
		if got := DdotIncSIMD(x, y, uintptr(n), 1, 1, 2, 3); got != ddot {
			t.Errorf("DdotIncSIMD n=%d: got %v want %v", n, got, ddot)
		}
		if got := SumSIMD(x[2 : 2+n]); got != sum {
			t.Errorf("SumSIMD n=%d: got %v want %v", n, got, sum)
		}
	}
	if got := DotIncSIMD(nil, nil, 0, 1, 1, 100, 100); got != 0 {
		t.Errorf("empty DotIncSIMD: got %v", got)
	}
	if got := DdotIncSIMD(nil, nil, 0, 1, 1, 100, 100); got != 0 {
		t.Errorf("empty DdotIncSIMD: got %v", got)
	}
}

func TestPortableSIMDAxpyIncrements(t *testing.T) {
	width := simd.BroadcastFloat32s(0).Len()
	for _, n := range []int{0, 1, width - 1, width, width + 1, 4*width + 1} {
		for _, inc := range []struct{ x, y, dst uintptr }{{1, 1, 1}, {2, 3, 2}, {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {2, 2, 2}} {
			for _, shared := range []bool{false, true} {
				x := make([]float32, 4*n+20)
				y := make([]float32, len(x))
				for i := range x {
					x[i], y[i] = float32(i%5-2), float32(i%7-3)
				}
				if shared {
					y = x
				}
				wantX, wantY := slices.Clone(x), slices.Clone(y)
				if shared {
					wantY = wantX
				}
				for i, ix, iy := 0, uintptr(1), uintptr(3); i < n; i++ {
					wantY[iy] += 0.5 * wantX[ix]
					ix, iy = ix+inc.x, iy+inc.y
				}
				AxpyIncSIMD(0.5, x, y, uintptr(n), inc.x, inc.y, 1, 3)
				if !slices.Equal(y, wantY) {
					t.Errorf("AxpyIncSIMD n=%d inc=%v shared=%t", n, inc, shared)
				}
				dst, wantDst := slices.Clone(y), slices.Clone(wantY)
				if shared {
					dst, wantDst = x, wantX
				}
				for i, ix, iy, idst := 0, uintptr(1), uintptr(3), uintptr(2); i < n; i++ {
					wantDst[idst] = 0.5*wantX[ix] + wantY[iy]
					ix, iy, idst = ix+inc.x, iy+inc.y, idst+inc.dst
				}
				AxpyIncToSIMD(dst, inc.dst, 2, 0.5, x, y, uintptr(n), inc.x, inc.y, 1, 3)
				if !slices.Equal(dst, wantDst) {
					t.Errorf("AxpyIncToSIMD n=%d inc=%v shared=%t", n, inc, shared)
				}
			}
		}
	}
	AxpyIncSIMD(1, nil, nil, 0, 1, 1, 100, 100)
	AxpyIncToSIMD(nil, 1, 100, 1, nil, nil, 0, 1, 1, 100, 100)
}

func TestPortableSIMDGerUnitStride(t *testing.T) {
	for _, n := range []int{0, 1, 3, 7, 16, 63, 64, 65} {
		const m = 5
		lda := n + 3
		x, y := []float32{1, -2, 3, -4, 5}, make([]float32, n)
		for i := range y {
			y[i] = float32(i%7 - 3)
		}
		a := make([]float32, m*lda)
		for i := range a {
			a[i] = 17
		}
		want := slices.Clone(a)
		for row := 0; row < m; row++ {
			for col := 0; col < n; col++ {
				want[row*lda+col] += 0.5 * x[row] * y[col]
			}
		}
		GerSIMD(m, uintptr(n), 0.5, x, 1, y, 1, a, uintptr(lda))
		if !slices.Equal(a, want) {
			t.Errorf("GerSIMD n=%d: got %v want %v", n, a, want)
		}
	}
}

func TestPortableSIMDScratchAllocations(t *testing.T) {
	x, y, dst := make([]float32, 256), make([]float32, 256), make([]float32, 256)
	for _, test := range []struct {
		name string
		fn   func()
	}{
		{"AxpyInc", func() { AxpyIncSIMD(0, x, y, 64, 2, 2, 0, 0) }},
		{"AxpyIncTo", func() { AxpyIncToSIMD(dst, 2, 0, 0, x, y, 64, 2, 2, 0, 0) }},
		{"DotUnitary", func() { DotUnitarySIMD(x, y) }},
		{"DotInc", func() { DotIncSIMD(x, y, 64, 2, 2, 0, 0) }},
		{"DdotUnitary", func() { DdotUnitarySIMD(x, y) }},
		{"DdotInc", func() { DdotIncSIMD(x, y, 64, 2, 2, 0, 0) }},
		{"Sum", func() { SumSIMD(x) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := testing.AllocsPerRun(100, test.fn); got != 0 {
				t.Fatalf("unexpected allocations: %v", got)
			}
		})
	}
}

func TestPortableSIMDBitStaging(t *testing.T) {
	values := []float32{0, math.Float32frombits(1 << 31), 1, -2, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), math.Float32frombits(0x7fc00123)}
	n := 3*simd.BroadcastFloat32s(0).Len() + 1
	for _, backward := range []bool{false, true} {
		x, y, dst := make([]float32, 2*n), make([]float32, 3*n), make([]float32, 4*n)
		for i := 0; i < n; i++ {
			x[2*i] = values[i%len(values)]
			y[3*i] = float32(i%3 - 1)
		}
		ix, iy, idst, incX, incY, incDst := uintptr(0), uintptr(0), uintptr(0), uintptr(2), uintptr(3), uintptr(4)
		if backward {
			ix, iy, idst = uintptr(2*(n-1)), uintptr(3*(n-1)), uintptr(4*(n-1))
			incX, incY, incDst = -incX, -incY, -incDst
		}
		want := slices.Clone(dst)
		wantY := slices.Clone(y)
		for i, jx, jy, jd := 0, ix, iy, idst; i < n; i++ {
			want[jd] = 0.5*x[jx] + y[jy]
			wantY[jy] += 0.5 * x[jx]
			jx, jy, jd = jx+incX, jy+incY, jd+incDst
		}
		AxpyIncToSIMD(dst, incDst, idst, 0.5, x, y, uintptr(n), incX, incY, ix, iy)
		AxpyIncSIMD(0.5, x, y, uintptr(n), incX, incY, ix, iy)
		for i, value := range dst {
			if math.Float32bits(value) != math.Float32bits(want[i]) && !(math.IsNaN(float64(value)) && math.IsNaN(float64(want[i]))) {
				t.Errorf("AxpyIncToSIMD backward=%t i=%d: got %v want %v", backward, i, value, want[i])
			}
		}
		for i, value := range y {
			if math.Float32bits(value) != math.Float32bits(wantY[i]) && !(math.IsNaN(float64(value)) && math.IsNaN(float64(wantY[i]))) {
				t.Errorf("AxpyIncSIMD backward=%t i=%d: got %v want %v", backward, i, value, wantY[i])
			}
		}
		if got := DotIncSIMD(x, y, uintptr(n), incX, incY, ix, iy); !math.IsNaN(float64(got)) {
			t.Errorf("DotIncSIMD backward=%t: got %v want NaN", backward, got)
		}
	}
}
