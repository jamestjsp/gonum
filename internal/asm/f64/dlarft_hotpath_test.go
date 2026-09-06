// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"testing"
)

var (
	dlarftAxpySink []float64
	dlarftGemvSink []float64
)

func TestAxpyIncDlarftGeometry(t *testing.T) {
	for _, n := range []int{0, 1, 2, 7, 8, 15, 16, 17, 31} {
		for _, stride := range []int{1, 32, 0, -32} {
			t.Run(fmt.Sprintf("n=%d/incy=%d", n, stride), func(t *testing.T) {
				x := make([]float64, n+4)
				for i := range x {
					x[i] = float64(i%7+1) / 8
				}
				y, iy := dlarftStridedVector(n, stride, 3)
				for i := range y {
					y[i] = float64(i%11-5) / 8
				}
				xOrig, want := slices.Clone(x), slices.Clone(y)
				dlarftAxpyReference(-0.25, x[2:], want, n, 1, stride, 0, iy)
				AxpyInc(-0.25, x[2:], y, uintptr(n), 1, uintptr(stride), 0, uintptr(iy))
				dlarftCheckBits(t, y, want)
				if !slices.Equal(x, xOrig) {
					t.Fatal("x changed")
				}
			})
		}
	}
}

func TestAxpyIncDlarftOverlap(t *testing.T) {
	if runtime.GOARCH == "amd64" {
		t.Skip("amd64 assembly does not guarantee partial-overlap semantics")
	}
	for _, tc := range []struct {
		name              string
		n, xoff, yoff, sy int
	}{
		{name: "forward-unit", n: 16, xoff: 1, yoff: 2, sy: 1},
		{name: "reverse-unit", n: 16, xoff: 2, yoff: 1, sy: 1},
		{name: "forward-ldt", n: 8, xoff: 1, yoff: 2, sy: 32},
		{name: "reverse-ldt", n: 8, xoff: 2, yoff: 1, sy: 32},
		{name: "forward-zero", n: 8, xoff: 1, yoff: 2, sy: 0},
		{name: "reverse-zero", n: 8, xoff: 2, yoff: 1, sy: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			data := make([]float64, max(tc.xoff+tc.n, tc.yoff+(tc.n-1)*tc.sy+1)+2)
			for i := range data {
				data[i] = float64(i+1) / 8
			}
			want := slices.Clone(data)
			dlarftAxpyReference(-0.25, want, want, tc.n, 1, tc.sy, tc.xoff, tc.yoff)
			AxpyInc(-0.25, data, data, uintptr(tc.n), 1, uintptr(tc.sy), uintptr(tc.xoff), uintptr(tc.yoff))
			dlarftCheckBits(t, data, want)
		})
	}
}

func TestAxpyIncDlarftShortX(t *testing.T) {
	if runtime.GOARCH == "amd64" {
		t.Skip("amd64 assembly does not guarantee bounds checks")
	}
	for _, stride := range []int{32, 0} {
		t.Run(fmt.Sprintf("incy=%d", stride), func(t *testing.T) {
			y, iy := dlarftStridedVector(2, stride, 0)
			for i := range y {
				y[i] = 5
			}
			defer func() {
				if recover() == nil {
					t.Fatal("short x did not panic")
				}
				if y[iy] != 5.5 {
					t.Fatalf("first update: got %g want 5.5", y[iy])
				}
				if stride != 0 && y[iy+stride] != 5 {
					t.Fatalf("second destination changed to %g", y[iy+stride])
				}
			}()
			AxpyInc(0.5, []float64{1}, y, 2, 1, uintptr(stride), 0, uintptr(iy))
		})
	}
	t.Run("immediate", func(t *testing.T) {
		defer func() {
			if recover() == nil {
				t.Fatal("empty x did not panic")
			}
		}()
		AxpyInc(0.5, nil, []float64{5}, 1, 1, 1, 0, 0)
	})
}

func TestAxpyIncDlarftExceptional(t *testing.T) {
	for _, tc := range []struct {
		name        string
		alpha, x, y float64
		check       func(float64) bool
	}{
		{name: "nan", alpha: 1, x: math.NaN(), y: 1, check: math.IsNaN},
		{name: "infinity", alpha: 1, x: math.Inf(1), y: 1, check: func(v float64) bool { return math.IsInf(v, 1) }},
		{name: "zero-times-infinity", alpha: 0, x: math.Inf(1), y: 1, check: math.IsNaN},
		{name: "negative-zero", alpha: math.Copysign(0, -1), x: 1, y: math.Copysign(0, -1), check: func(v float64) bool { return math.Float64bits(v) == 1<<63 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			y := []float64{tc.y}
			AxpyInc(tc.alpha, []float64{tc.x}, y, 1, 1, 1, 0, 0)
			if !tc.check(y[0]) {
				t.Fatalf("unexpected result %g", y[0])
			}
		})
	}
}

func TestGemvTDlarftGeometry(t *testing.T) {
	for _, tc := range []struct {
		m, n, lda, incX, incY int
	}{
		{65, 1, 128, 128, 32},
		{65, 7, 128, 128, 32},
		{129, 16, 256, 256, 32},
		{257, 31, 256, 256, 32},
		{65, 17, 20, 1, 1},
	} {
		t.Run(fmt.Sprintf("m=%d/n=%d/lda=%d/incx=%d/incy=%d", tc.m, tc.n, tc.lda, tc.incX, tc.incY), func(t *testing.T) {
			a := make([]float64, tc.m*tc.lda)
			for i := range a {
				a[i] = float64(i%17-8) / 16
			}
			x, ix := dlarftStridedVector(tc.m, tc.incX, 2)
			y, iy := dlarftStridedVector(tc.n, tc.incY, 3)
			for i := range x {
				x[i] = float64(i%13+1) / 16
			}
			for i := range y {
				y[i] = math.NaN()
			}
			aOrig, xOrig, want := slices.Clone(a), slices.Clone(x), slices.Clone(y)
			dlarftGemvTReference(tc.m, tc.n, -0.75, a, tc.lda, x, tc.incX, ix, 0, want, tc.incY, iy)
			GemvT(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(tc.lda), x[ix:], uintptr(tc.incX), 0, y[iy:], uintptr(tc.incY))
			dlarftCheckClose(t, y, want, 32*0x1p-52*float64(tc.m))
			if !slices.Equal(a, aOrig) || !slices.Equal(x, xOrig) {
				t.Fatal("read-only input changed")
			}
		})
	}
}

func BenchmarkAxpyIncDlarft(b *testing.B) {
	for _, n := range []int{1, 2, 7, 8, 15, 16, 17, 31} {
		for _, stride := range []int{1, 32, 0, -32} {
			b.Run(fmt.Sprintf("n=%d/incy=%d", n, stride), func(b *testing.B) {
				x := make([]float64, n)
				y, iy := dlarftStridedVector(n, stride, 0)
				for i := range x {
					x[i] = float64(i%7+1) / 8
				}
				for i := range y {
					y[i] = float64(i%5+1) / 8
				}
				alpha := 0.25
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					AxpyInc(alpha, x, y, uintptr(n), 1, uintptr(stride), 0, uintptr(iy))
					alpha = -alpha
				}
				b.StopTimer()
				dlarftAxpySink = y
			})
		}
	}
}

func BenchmarkGemvTDlarft(b *testing.B) {
	for _, tc := range []struct {
		name                  string
		m, n, lda, incX, incY int
	}{
		{name: "dlarft-128x1", m: 128, n: 1, lda: 128, incX: 128, incY: 32},
		{name: "dlarft-128x7", m: 128, n: 7, lda: 128, incX: 128, incY: 32},
		{name: "dlarft-128x16", m: 128, n: 16, lda: 128, incX: 128, incY: 32},
		{name: "dlarft-128x31", m: 128, n: 31, lda: 128, incX: 128, incY: 32},
		{name: "dlarft-256x1", m: 256, n: 1, lda: 256, incX: 256, incY: 32},
		{name: "dlarft-256x16", m: 256, n: 16, lda: 256, incX: 256, incY: 32},
		{name: "dlarft-256x31", m: 256, n: 31, lda: 256, incX: 256, incY: 32},
		{name: "contiguous-128x16", m: 128, n: 16, lda: 16, incX: 1, incY: 1},
		{name: "contiguous-256x31", m: 256, n: 31, lda: 31, incX: 1, incY: 1},
	} {
		b.Run(tc.name, func(b *testing.B) {
			a := make([]float64, tc.m*tc.lda)
			x, _ := dlarftStridedVector(tc.m, tc.incX, 0)
			y, _ := dlarftStridedVector(tc.n, tc.incY, 0)
			for i := range a {
				a[i] = float64(i%17-8) / 16
			}
			for i := range x {
				x[i] = float64(i%13+1) / 16
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GemvT(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(tc.lda), x, uintptr(tc.incX), 0, y, uintptr(tc.incY))
			}
			b.StopTimer()
			dlarftGemvSink = y
		})
	}
}

func dlarftStridedVector(n, stride, guard int) ([]float64, int) {
	if n == 0 {
		return make([]float64, 2*guard), guard
	}
	step := stride
	if step < 0 {
		step = -step
	}
	data := make([]float64, 2*guard+(n-1)*step+1)
	start := guard
	if stride < 0 {
		start += (n - 1) * step
	}
	return data, start
}

func dlarftAxpyReference(alpha float64, x, y []float64, n, incX, incY, ix, iy int) {
	for i := 0; i < n; i++ {
		y[iy] += alpha * x[ix]
		ix += incX
		iy += incY
	}
}

func dlarftGemvTReference(m, n int, alpha float64, a []float64, lda int, x []float64, incX, ix int, beta float64, y []float64, incY, iy int) {
	for j, jy := 0, iy; j < n; j, jy = j+1, jy+incY {
		if beta == 0 {
			y[jy] = 0
		} else {
			y[jy] *= beta
		}
	}
	for i := 0; i < m; i++ {
		for j, jy := 0, iy; j < n; j, jy = j+1, jy+incY {
			y[jy] += alpha * x[ix] * a[i*lda+j]
		}
		ix += incX
	}
}

func dlarftCheckBits(t *testing.T, got, want []float64) {
	t.Helper()
	for i := range got {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) && !(math.IsNaN(got[i]) && math.IsNaN(want[i])) {
			t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
		}
	}
}

func dlarftCheckClose(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	for i := range got {
		if math.IsNaN(want[i]) {
			if !math.IsNaN(got[i]) {
				t.Fatalf("index %d: guard changed to %g", i, got[i])
			}
			continue
		}
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) || math.Abs(got[i]-want[i]) > tol*(1+math.Abs(want[i])) {
			t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
		}
	}
}
