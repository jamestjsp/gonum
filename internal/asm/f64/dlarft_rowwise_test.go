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

var dlarftRowWiseSink []float64

func TestGemvNRowWiseDlarftGeometry(t *testing.T) {
	for _, tc := range []struct {
		m, n, incY, offset int
		beta               float64
	}{
		{1, 7, 32, 0, 0}, {2, 8, 64, 1, 1}, {3, 9, 32, 3, -0.5},
		{4, 7, 32, 1, 0}, {4, 8, 64, 3, 1}, {4, 9, 32, 0, -0.5},
		{4, 15, 64, 0, 1}, {5, 16, 32, 1, 0}, {7, 17, 64, 3, -0.5},
		{8, 31, 32, 0, 1}, {31, 32, 64, 1, 0}, {32, 33, 32, 3, -0.5},
		{33, 64, 64, 0, 1}, {4, 128, 32, 1, 0}, {32, 256, 64, 3, -0.5},
	} {
		t.Run(fmt.Sprintf("m=%d/n=%d/incy=%d/offset=%d/beta=%g", tc.m, tc.n, tc.incY, tc.offset, tc.beta), func(t *testing.T) {
			const guard = 4
			lda := tc.n + 5
			vStore := dlarftRowWiseData(guard + tc.offset + (tc.m+1)*lda + guard)
			yStore := dlarftRowWiseData(guard + (tc.m-1)*tc.incY + 1 + guard)
			v := vStore[guard+tc.offset : len(vStore)-guard]
			a, x := v[lda:], v
			y := yStore[guard : len(yStore)-guard]
			if tc.beta == 0 {
				for i := 0; i < tc.m; i++ {
					y[i*tc.incY] = math.NaN()
				}
			}
			vOrig := slices.Clone(vStore)
			want := slices.Clone(yStore)
			dlarftRowWiseReference(tc.m, tc.n, -0.75, a, lda, x, tc.beta, want[guard:], tc.incY)
			GemvN(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(lda), x, 1, tc.beta, y, uintptr(tc.incY))
			dlarftRowWiseCheck(t, yStore, want, 32*0x1p-52*float64(tc.n))
			if !slices.Equal(vStore, vOrig) {
				t.Fatal("shared A/X storage changed")
			}
		})
	}
}

func TestGemvNRowWiseDlarftSequentialParity(t *testing.T) {
	for _, tc := range []struct {
		m, n, incY int
		beta       float64
	}{
		{4, 8, 2, 0}, {7, 9, 64, 1}, {31, 17, 2, -0.5},
		{32, 33, 64, 0}, {4, 257, 64, 1},
	} {
		t.Run(fmt.Sprintf("m=%d/n=%d/incy=%d/beta=%g", tc.m, tc.n, tc.incY, tc.beta), func(t *testing.T) {
			const guard = 3
			lda := tc.n + 3
			a := dlarftRowWiseParityData(tc.m * lda)
			x := dlarftRowWiseParityData(tc.n)
			yStore := dlarftRowWiseParityData(guard + (tc.m-1)*tc.incY + 1 + guard)
			y := yStore[guard : len(yStore)-guard]
			if tc.beta == 0 {
				for i := 0; i < tc.m; i++ {
					y[i*tc.incY] = math.NaN()
				}
			}
			want := slices.Clone(yStore)
			gemvN(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(lda), x, 1, tc.beta, want[guard:], uintptr(tc.incY))
			GemvN(uintptr(tc.m), uintptr(tc.n), -0.75, a, uintptr(lda), x, 1, tc.beta, y, uintptr(tc.incY))
			dlarftRowWiseCheckBits(t, yStore, want)
		})
	}
}

func TestGemvNRowWiseDlarftActiveAXOverlap(t *testing.T) {
	const m, n, lda, incY = 4, 9, 11, 64
	storage := dlarftRowWiseData(m * lda)
	orig := slices.Clone(storage)
	y := dlarftRowWiseData((m-1)*incY + 1)
	want := slices.Clone(y)
	dlarftRowWiseReference(m, n, -0.75, orig, lda, orig, -0.5, want, incY)
	GemvN(m, n, -0.75, storage, lda, storage, 1, -0.5, y, incY)
	dlarftRowWiseCheck(t, y, want, 32*0x1p-52*n)
	if !slices.Equal(storage, orig) {
		t.Fatal("overlapping A/X storage changed")
	}
}

func TestGemvNRowWiseDlarftExceptional(t *testing.T) {
	t.Run("coefficients", func(t *testing.T) {
		const m, n, lda, incY = 4, 8, 10, 3
		for _, value := range []float64{0, math.Inf(1), math.NaN()} {
			a := make([]float64, m*lda)
			x := make([]float64, n)
			y := dlarftRowWiseData((m-1)*incY + 1)
			for i := range x {
				x[i] = 1
			}
			for i := 0; i < m; i++ {
				a[i*lda] = value
			}
			want := slices.Clone(y)
			dlarftRowWiseReference(m, n, 1, a, lda, x, 0, want, incY)
			GemvN(m, n, 1, a, lda, x, 1, 0, y, incY)
			dlarftRowWiseCheck(t, y, want, 0)
		}
	})
	t.Run("ordered-overflow-cancellation", func(t *testing.T) {
		const m, n, lda, incY = 4, 8, 8, 32
		a := make([]float64, m*lda)
		x := make([]float64, n)
		x[0], x[1], x[2] = 1, 1, 1
		a[0], a[1], a[2] = math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64
		a[lda], a[lda+1], a[lda+2] = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64
		y := make([]float64, (m-1)*incY+1)
		GemvN(m, n, 1, a, lda, x, 1, 0, y, incY)
		if !math.IsInf(y[0], 1) || y[incY] != math.MaxFloat64 {
			t.Fatalf("got [%g %g] want [+Inf %g]", y[0], y[incY], math.MaxFloat64)
		}
	})
	t.Run("signed-zero", func(t *testing.T) {
		a := make([]float64, 4*8)
		x := make([]float64, 8)
		for i := range a {
			a[i] = 1
		}
		for i := range x {
			x[i] = 1
		}
		y := make([]float64, 10)
		y[0] = math.Copysign(0, -1)
		GemvN(4, 8, math.Copysign(0, -1), a, 8, x, 1, 1, y, 3)
		if math.Float64bits(y[0]) != math.Float64bits(math.Copysign(0, -1)) {
			t.Fatalf("got bits %#x want negative zero", math.Float64bits(y[0]))
		}
	})
}

func TestGemvNRowWiseDlarftOverlapFallback(t *testing.T) {
	for _, matrix := range []bool{false, true} {
		t.Run(fmt.Sprintf("matrix=%t", matrix), func(t *testing.T) {
			const m, n, lda, incY = 5, 9, 11, 3
			got := dlarftRowWiseData(96)
			want := slices.Clone(got)
			a, wa := dlarftRowWiseData(m*lda), dlarftRowWiseData(m*lda)
			x, wx := got[2:], want[2:]
			yoff := 3
			if matrix {
				a, wa = got, want
				x, wx = dlarftRowWiseData(n), dlarftRowWiseData(n)
				yoff = 4
			}
			gemvN(m, n, -0.75, wa, lda, wx, 1, -0.5, want[yoff:], incY)
			GemvN(m, n, -0.75, a, lda, x, 1, -0.5, got[yoff:], incY)
			dlarftRowWiseCheck(t, got, want, 0)
		})
	}
}

func TestGemvNRowWiseDlarftIncrementFallback(t *testing.T) {
	for _, tc := range []struct{ incX, incY int }{
		{0, 64}, {-1, 64}, {2, 64}, {1, 0}, {1, -3},
	} {
		t.Run(fmt.Sprintf("incx=%d/incy=%d", tc.incX, tc.incY), func(t *testing.T) {
			const m, n, lda = 5, 9, 11
			a := dlarftRowWiseData(m * lda)
			x := dlarftRowWiseVector(n, tc.incX)
			y := dlarftRowWiseVector(m, tc.incY)
			want := slices.Clone(y)
			gemvN(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, want, uintptr(tc.incY))
			GemvN(m, n, -0.75, a, lda, x, uintptr(tc.incX), -0.5, y, uintptr(tc.incY))
			dlarftRowWiseCheckBits(t, y, want)
		})
	}
}

func TestGemvNRowWiseDlarftInvalidFallback(t *testing.T) {
	const m, n, lda, incY = 5, 9, 11, 3
	yLen := (m-1)*incY + 1
	for _, tc := range []struct {
		name    string
		a, x, y []float64
	}{
		{name: "short-a", a: dlarftRowWiseData((m-1)*lda + n - 1), x: dlarftRowWiseData(n), y: dlarftRowWiseData(yLen)},
		{name: "short-x", a: dlarftRowWiseData(m * lda), x: dlarftRowWiseData(n - 1), y: dlarftRowWiseData(yLen)},
		{name: "short-y", a: dlarftRowWiseData(m * lda), x: dlarftRowWiseData(n), y: dlarftRowWiseData(yLen - 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ga, gx, gy := slices.Clone(tc.a), slices.Clone(tc.x), slices.Clone(tc.y)
			wa, wx, wy := slices.Clone(tc.a), slices.Clone(tc.x), slices.Clone(tc.y)
			ga, gx, gy = ga[:len(ga):len(ga)], gx[:len(gx):len(gx)], gy[:len(gy):len(gy)]
			wa, wx, wy = wa[:len(wa):len(wa)], wx[:len(wx):len(wx)], wy[:len(wy):len(wy)]
			gpanic := dlarftRowWiseRun(func() { GemvN(m, n, -0.75, ga, lda, gx, 1, 1, gy, incY) })
			wpanic := dlarftRowWiseRun(func() { gemvN(m, n, -0.75, wa, lda, wx, 1, 1, wy, incY) })
			if !gpanic || !wpanic {
				t.Fatalf("panic: got %t want fallback %t", gpanic, wpanic)
			}
			dlarftRowWiseCheck(t, ga, wa, 0)
			dlarftRowWiseCheck(t, gx, wx, 0)
			dlarftRowWiseCheck(t, gy, wy, 0)
		})
	}
}

func BenchmarkGemvNRowWiseDlarft(b *testing.B) {
	for _, tc := range []struct{ m, n, incX, incY int }{
		{3, 8, 1, 32}, {4, 7, 1, 32}, {4, 8, 1, 32}, {5, 9, 1, 64},
		{31, 15, 1, 32}, {31, 16, 1, 64}, {32, 17, 1, 32}, {33, 33, 1, 64},
		{4, 128, 1, 64}, {32, 256, 1, 64},
		{31, 128, 1, 32}, {31, 256, 1, 64}, {31, 512, 1, 64},
		{4, 128, 1, 1}, {8, 128, 1, 1}, {4, 128, 2, 64}, {33, 128, 1, 64},
	} {
		b.Run(fmt.Sprintf("m=%d/n=%d/incx=%d/incy=%d", tc.m, tc.n, tc.incX, tc.incY), func(b *testing.B) {
			lda := tc.n + 5
			v := dlarftRowWiseData((tc.m + 1) * lda)
			x := v
			if tc.incX != 1 {
				x = dlarftRowWiseData((tc.n-1)*tc.incX + 1)
			}
			y := make([]float64, (tc.m-1)*tc.incY+1)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				GemvN(uintptr(tc.m), uintptr(tc.n), -0.75, v[lda:], uintptr(lda), x, uintptr(tc.incX), 0, y, uintptr(tc.incY))
			}
			b.StopTimer()
			dlarftRowWiseSink = y
		})
	}
}

func dlarftRowWiseReference(m, n int, alpha float64, a []float64, lda int, x []float64, beta float64, y []float64, incY int) {
	for i := 0; i < m; i++ {
		dot := 0.0
		for j := 0; j < n; j++ {
			dot += a[i*lda+j] * x[j]
		}
		if beta == 0 {
			y[i*incY] = alpha * dot
		} else {
			y[i*incY] = beta*y[i*incY] + alpha*dot
		}
	}
}

func dlarftRowWiseData(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%17-8) / 16
	}
	return x
}

func dlarftRowWiseParityData(n int) []float64 {
	x := dlarftRowWiseData(n)
	for i := range x {
		switch i % 29 {
		case 0:
			x[i] = math.SmallestNonzeroFloat64
		case 1:
			x[i] = -math.SmallestNonzeroFloat64
		case 2:
			x[i] = 0
		case 3:
			x[i] = math.Copysign(0, -1)
		}
	}
	return x
}

func dlarftRowWiseVector(n, inc int) []float64 {
	step := inc
	if step < 0 {
		step = -step
	}
	return dlarftRowWiseParityData((n-1)*step + 1)
}

func dlarftRowWiseCheckBits(t *testing.T, got, want []float64) {
	t.Helper()
	for i := range got {
		if math.IsNaN(want[i]) {
			if !math.IsNaN(got[i]) {
				t.Fatalf("index %d: got %g want NaN", i, got[i])
			}
			continue
		}
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("index %d: got %g (%#x) want %g (%#x)", i, got[i], math.Float64bits(got[i]), want[i], math.Float64bits(want[i]))
		}
	}
}

func dlarftRowWiseCheck(t *testing.T, got, want []float64, tol float64) {
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

func dlarftRowWiseRun(fn func()) (panicked bool) {
	defer func() { panicked = recover() != nil }()
	fn()
	return false
}
