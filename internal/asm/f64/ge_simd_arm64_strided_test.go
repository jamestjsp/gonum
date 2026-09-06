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

var gemvTArm64StridedSink []float64

func TestGemvTArm64Strided(t *testing.T) {
	cases := []struct {
		m, n, incX, incY int
		beta             float64
	}{
		{7, 1, 1, 2, 0}, {8, 2, 2, 3, 1}, {9, 3, 128, 32, -0.5},
		{7, 4, 2, 64, 1}, {8, 7, 128, 32, 0}, {9, 8, 1, 64, -0.5},
		{7, 15, 128, 3, -0.5}, {8, 16, 1, 32, 1}, {9, 17, 2, 64, 0},
		{8, 31, 128, 32, -0.5}, {9, 32, 1, 64, 1}, {8, 33, 2, 3, 0},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("m=%d/n=%d/incx=%d/incy=%d/beta=%g", tc.m, tc.n, tc.incX, tc.incY, tc.beta), func(t *testing.T) {
			const guard = 3
			lda := max(tc.n+3, 128)
			aStore := gemvTArm64Data(guard + tc.m*lda + guard)
			xStore := gemvTArm64Data(guard + (tc.m-1)*tc.incX + 1 + guard)
			yStore := gemvTArm64Data(guard + (tc.n-1)*tc.incY + 1 + guard)
			a := aStore[guard : len(aStore)-guard]
			x := xStore[guard : len(xStore)-guard]
			y := yStore[guard : len(yStore)-guard]
			if tc.beta == 0 {
				for j := 0; j < tc.n; j++ {
					y[j*tc.incY] = math.NaN()
				}
			}
			aOrig, xOrig := slices.Clone(aStore), slices.Clone(xStore)
			want := slices.Clone(yStore)
			gemvTArm64Sequential(tc.m, tc.n, -0.75, a, lda, x, tc.incX, tc.beta, want[guard:], tc.incY)
			GemvT(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(lda), x, uintptr(tc.incX), tc.beta, y, uintptr(tc.incY))
			gemvTArm64Check(t, yStore, want, 32*0x1p-52*float64(tc.m))
			if !slices.Equal(aStore, aOrig) || !slices.Equal(xStore, xOrig) {
				t.Fatal("read-only input changed")
			}
		})
	}
}

func TestGemvTArm64StridedExceptional(t *testing.T) {
	t.Run("coefficients", func(t *testing.T) {
		const incY = 3
		for _, value := range []float64{0, math.Inf(1), math.NaN()} {
			a := make([]float64, 8*5)
			x := make([]float64, 8)
			y := gemvTArm64Data((3-1)*incY + 1)
			for i := range x {
				x[i] = 1
			}
			y[0], y[incY], y[2*incY] = 1, 2, 3
			a[0], a[1], a[2] = value, value, value
			want := slices.Clone(y)
			gemvTArm64Sequential(8, 3, 1, a, 5, x, 1, 0, want, incY)
			GemvT(8, 3, 1, a, 5, x, 1, 0, y, incY)
			gemvTArm64Check(t, y, want, 0)
		}
	})
	t.Run("ordered-overflow-cancellation", func(t *testing.T) {
		const incY = 32
		a := make([]float64, 8*2)
		x := make([]float64, 8)
		for i := range x {
			x[i] = 1
		}
		a[0], a[2], a[4] = math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		a[1], a[3], a[5] = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64
		y := make([]float64, incY+1)
		y[0], y[incY] = math.NaN(), math.NaN()
		GemvT(8, 2, 1, a, 2, x, 1, 0, y, incY)
		if !math.IsInf(y[0], 1) || y[incY] != math.MaxFloat64 {
			t.Fatalf("got [%g %g] want [+Inf %g]", y[0], y[incY], math.MaxFloat64)
		}
	})
	t.Run("signed-zero", func(t *testing.T) {
		a := make([]float64, 8)
		x := make([]float64, 8)
		for i := range a {
			a[i], x[i] = 1, 1
		}
		y := []float64{math.Copysign(0, -1)}
		GemvT(8, 1, math.Copysign(0, -1), a, 1, x, 1, 1, y, 3)
		if math.Float64bits(y[0]) != math.Float64bits(math.Copysign(0, -1)) {
			t.Fatalf("got bits %#x want negative zero", math.Float64bits(y[0]))
		}
	})
}

func TestGemvTArm64StridedSharedAX(t *testing.T) {
	const m, n, lda, xoff, incY = 9, 7, 16, 7, 64
	storage := gemvTArm64Data(m * lda)
	orig := slices.Clone(storage)
	y := gemvTArm64Data((n-1)*incY + 1)
	want := slices.Clone(y)
	gemvTArm64Sequential(m, n, -0.75, orig, lda, orig[xoff:], lda, -0.5, want, incY)
	GemvT(m, n, -0.75, storage, lda, storage[xoff:], lda, -0.5, y, incY)
	gemvTArm64Check(t, y, want, 32*0x1p-52*m)
	if !slices.Equal(storage, orig) {
		t.Fatal("shared A/X storage changed")
	}
}

func TestGemvTArm64StridedOverlapFallback(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		matrix                 bool
		incX, incY, xoff, yoff int
	}{
		{name: "x-y-ldt32", incX: 1, incY: 32, xoff: 2, yoff: 3},
		{name: "x-y-ldt64", incX: 1, incY: 64, xoff: 3, yoff: 2},
		{name: "a-y", matrix: true, incX: 1, incY: 3, xoff: 0, yoff: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const m, n, lda = 9, 7, 10
			length := max(m*lda+8, max(tc.xoff+(m-1)*tc.incX+1, tc.yoff+(n-1)*tc.incY+1)+4)
			got := gemvTArm64Data(length)
			want := slices.Clone(got)
			a, wa := gemvTArm64Data(m*lda), gemvTArm64Data(m*lda)
			x, wx := got[tc.xoff:], want[tc.xoff:]
			if tc.matrix {
				a, wa = got, want
				x, wx = gemvTArm64Data(m), gemvTArm64Data(m)
			}
			gemvT(m, n, -0.75, wa, lda, wx, uintptr(tc.incX), -0.5, want[tc.yoff:], uintptr(tc.incY))
			GemvT(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, got[tc.yoff:], uintptr(tc.incY))
			gemvTArm64Check(t, got, want, 0)
		})
	}
}

func TestGemvTArm64StridedIncrementFallback(t *testing.T) {
	for _, tc := range []struct {
		incX, incY int
	}{
		{0, 32}, {-2, 64}, {2, 0}, {2, -3},
	} {
		t.Run(fmt.Sprintf("incx=%d/incy=%d", tc.incX, tc.incY), func(t *testing.T) {
			const m, n, lda = 9, 7, 10
			x, _ := gemvTArm64Vector(m, tc.incX)
			y, _ := gemvTArm64Vector(n, tc.incY)
			a := gemvTArm64Data(m * lda)
			want := slices.Clone(y)
			gemvT(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, want, uintptr(tc.incY))
			GemvT(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, y, uintptr(tc.incY))
			gemvTArm64Check(t, y, want, 0)
		})
	}
}

func TestGemvTArm64StridedInvalidFallback(t *testing.T) {
	const m, n, lda, incY = 9, 7, 10, 3
	yLen := (n-1)*incY + 1
	for _, tc := range []struct {
		name    string
		a, x, y []float64
	}{
		{name: "short-a", a: gemvTArm64Data((m-1)*lda + n - 1), x: gemvTArm64Data(m), y: gemvTArm64Data(yLen)},
		{name: "short-x", a: gemvTArm64Data(m * lda), x: gemvTArm64Data(m - 1), y: gemvTArm64Data(yLen)},
		{name: "short-y", a: gemvTArm64Data(m * lda), x: gemvTArm64Data(m), y: gemvTArm64Data(yLen - 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ga, gx, gy := slices.Clone(tc.a), slices.Clone(tc.x), slices.Clone(tc.y)
			wa, wx, wy := slices.Clone(tc.a), slices.Clone(tc.x), slices.Clone(tc.y)
			ga, wa = ga[:len(ga):len(ga)], wa[:len(wa):len(wa)]
			gpanic := gemvTArm64Run(func() { GemvT(m, n, -0.75, ga, lda, gx, 1, 1, gy, incY) })
			wpanic := gemvTArm64Run(func() { gemvT(m, n, -0.75, wa, lda, wx, 1, 1, wy, incY) })
			if !gpanic || !wpanic {
				t.Fatalf("panic: got %t want fallback %t", gpanic, wpanic)
			}
			gemvTArm64Check(t, ga, wa, 0)
			gemvTArm64Check(t, gx, wx, 0)
			gemvTArm64Check(t, gy, wy, 0)
		})
	}
}

func BenchmarkGemvTArm64StridedBoundary(b *testing.B) {
	for _, tc := range []struct {
		m, n, incX, incY int
	}{
		{7, 16, 1, 32}, {8, 16, 1, 32}, {9, 16, 1, 32},
		{8, 1, 1, 64}, {8, 7, 2, 64}, {8, 31, 128, 32},
		{8, 32, 128, 64}, {8, 33, 128, 64},
	} {
		b.Run(fmt.Sprintf("m=%d/n=%d/incx=%d/incy=%d", tc.m, tc.n, tc.incX, tc.incY), func(b *testing.B) {
			lda := max(128, tc.n)
			a := gemvTArm64Data(tc.m * lda)
			x := gemvTArm64Data((tc.m-1)*tc.incX + 1)
			y := make([]float64, (tc.n-1)*tc.incY+1)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GemvT(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(lda), x, uintptr(tc.incX), 0, y, uintptr(tc.incY))
			}
			b.StopTimer()
			gemvTArm64StridedSink = y
		})
	}
}

func gemvTArm64Sequential(m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	for j := 0; j < n; j++ {
		if beta == 0 {
			y[j*incY] = 0
		} else {
			y[j*incY] *= beta
		}
	}
	for i := 0; i < m; i++ {
		scale := alpha * x[i*incX]
		for j := 0; j < n; j++ {
			y[j*incY] += scale * a[i*lda+j]
		}
	}
}

func gemvTArm64Data(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%17-8) / 16
	}
	return x
}

func gemvTArm64Vector(n, inc int) ([]float64, int) {
	step := inc
	if step < 0 {
		step = -step
	}
	x := gemvTArm64Data((n-1)*step + 1)
	if inc < 0 {
		return x, (n - 1) * step
	}
	return x, 0
}

func gemvTArm64Check(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	for i := range got {
		if math.IsNaN(want[i]) {
			if !math.IsNaN(got[i]) {
				t.Fatalf("index %d: got %g want NaN", i, got[i])
			}
			continue
		}
		if math.IsInf(want[i], 0) {
			if !math.IsInf(got[i], 0) || math.Signbit(got[i]) != math.Signbit(want[i]) {
				t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
			}
			continue
		}
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) || math.Abs(got[i]-want[i]) > tol*(1+math.Abs(want[i])) {
			t.Fatalf("index %d: got %g want %g", i, got[i], want[i])
		}
	}
}

func gemvTArm64Run(fn func()) (panicked bool) {
	defer func() {
		panicked = recover() != nil
	}()
	fn()
	return false
}
