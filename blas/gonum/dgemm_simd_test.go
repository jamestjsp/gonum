// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"simd"
	"slices"
	"testing"
)

func TestDgemmSIMDTiles(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 37))
	width := simd.BroadcastFloat64s(0).Len()
	tile := dgemmSIMDCols * width
	shapes := [][3]int{
		{dgemmSIMDRows - 1, tile, 4},
		{dgemmSIMDRows, tile - 1, 4},
		{dgemmSIMDRows, tile, 3},
		{dgemmSIMDRows, tile, 4},
		{dgemmSIMDRows, tile + 1, 5},
		{dgemmSIMDRows + 1, 2*tile - 1, 7},
		{2*dgemmSIMDRows - 1, 2*tile + 1, 17},
		{2 * dgemmSIMDRows, 2 * tile, 33},
		{4*dgemmSIMDRows + 1, 4*tile + 1, 31},
		{16*dgemmSIMDRows + 1, 2*tile + 1, 32},
	}
	for tail := 0; tail < tile; tail++ {
		shapes = append(shapes, [3]int{dgemmSIMDRows + 1, tile + tail, 9})
	}
	for _, dims := range shapes {
		m, n, k := dims[0], dims[1], dims[2]
		for _, trans := range []bool{false, true} {
			ar, ac := m, k
			if trans {
				ar, ac = k, m
			}
			lda, ldb, ldc := ac+3, n+3, n+5
			a, b, c := make([]float64, (ar-1)*lda+ac), make([]float64, (k-1)*ldb+n), make([]float64, (m-1)*ldc+n)
			for _, x := range [][]float64{a, b, c} {
				for i := range x {
					x[i] = rng.NormFloat64()
				}
			}
			want, cOrig := slices.Clone(c), slices.Clone(c)
			aOrig, bOrig := slices.Clone(a), slices.Clone(b)
			const alpha = -0.75
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					for l := 0; l < k; l++ {
						ai := i*lda + l
						if trans {
							ai = l*lda + i
						}
						want[i*ldc+j] += alpha * a[ai] * b[l*ldb+j]
					}
				}
			}
			accepted := dgemmSerialSIMD(trans, false, m, n, k, a, lda, b, ldb, c, ldc, alpha)
			wantAccepted := m >= dgemmSIMDRows && n >= tile && k >= 4
			if accepted != wantAccepted {
				t.Fatalf("dims=%v trans=%t accepted=%t want=%t", dims, trans, accepted, wantAccepted)
			}
			if !dgemmSIMDEqualBits(a, aOrig) || !dgemmSIMDEqualBits(b, bOrig) {
				t.Fatalf("dims=%v trans=%t input changed", dims, trans)
			}
			if !accepted {
				if !dgemmSIMDEqualBits(c, cOrig) {
					t.Fatalf("dims=%v trans=%t rejected kernel changed C", dims, trans)
				}
				continue
			}
			for i, v := range want {
				if i%ldc >= n {
					if math.Float64bits(c[i]) != math.Float64bits(v) {
						t.Fatalf("dims=%v trans=%t padding changed at %d", dims, trans, i)
					}
					continue
				}
				if !gemmSIMDClose(c[i], v, 1e-12, 1e-12) {
					t.Fatalf("dims=%v trans=%t index=%d got=%g want=%g", dims, trans, i, c[i], v)
				}
			}
		}
	}
}

func TestDgemmSIMDReject(t *testing.T) {
	m := dgemmSIMDRows + 1
	n := dgemmSIMDCols*simd.BroadcastFloat64s(0).Len() + 1
	k := n
	for axis := 0; axis < 3; axis++ {
		dims := [3]int{m, n, k}
		dims[axis] = 0
		c := []float64{1, -2, 3}
		if dgemmSerialSIMD(false, false, dims[0], dims[1], dims[2], nil, k, nil, n, c, n, 1) {
			t.Fatalf("accepted empty dimensions %v", dims)
		}
		if !slices.Equal(c, []float64{1, -2, 3}) {
			t.Fatalf("rejected dimensions %v changed C", dims)
		}
	}
	for _, trans := range []bool{false, true} {
		for _, alias := range []string{"transposed-b", "c=a", "c=a+1", "a=c+1", "c=b", "c=b+1", "b=c+1"} {
			t.Run(fmt.Sprintf("trans=%t/%s", trans, alias), func(t *testing.T) {
				lda := k
				if trans {
					lda = m
				}
				a, b, c := make([]float64, m*k), make([]float64, k*n), make([]float64, m*n)
				shared := make([]float64, max(len(a), len(b), len(c))+1)
				switch alias {
				case "c=a":
					a, c = shared[:len(a)], shared[:len(c)]
				case "c=a+1":
					a, c = shared[:len(a)], shared[1:len(c)+1]
				case "a=c+1":
					a, c = shared[1:len(a)+1], shared[:len(c)]
				case "c=b":
					b, c = shared[:len(b)], shared[:len(c)]
				case "c=b+1":
					b, c = shared[:len(b)], shared[1:len(c)+1]
				case "b=c+1":
					b, c = shared[1:len(b)+1], shared[:len(c)]
				}
				for _, x := range [][]float64{a, b, c} {
					for i := range x {
						x[i] = float64(i%13) - 6
					}
				}
				aOrig, bOrig, cOrig := slices.Clone(a), slices.Clone(b), slices.Clone(c)
				if dgemmSerialSIMD(trans, alias == "transposed-b", m, n, k, a, lda, b, n, c, n, 0.75) {
					t.Fatal("kernel accepted unsupported layout")
				}
				if !dgemmSIMDEqualBits(a, aOrig) || !dgemmSIMDEqualBits(b, bOrig) || !dgemmSIMDEqualBits(c, cOrig) {
					t.Fatal("rejected kernel changed an operand")
				}
			})
		}
	}
}

func TestDgemmSIMDZeroCoefficients(t *testing.T) {
	m, k := dgemmSIMDRows+1, 5
	n := 2*dgemmSIMDCols*simd.BroadcastFloat64s(0).Len() - 1
	for _, trans := range []bool{false, true} {
		for _, name := range []string{"all-zero", "mixed", "alpha-zero", "underflow"} {
			t.Run(fmt.Sprintf("trans=%t/%s", trans, name), func(t *testing.T) {
				lda := k
				if trans {
					lda = m
				}
				a, b, c := make([]float64, m*k), make([]float64, k*n), make([]float64, m*n)
				for l := 0; l < k; l++ {
					for j := 0; j < n; j++ {
						b[l*n+j] = math.NaN()
						if l%2 != 0 {
							b[l*n+j] = math.Inf(1)
						}
						if name == "mixed" && (l == 2 || l == 3) {
							b[l*n+j] = float64((j+l)%7) - 3
						}
					}
				}
				for i := 0; i < m; i++ {
					for l := 0; l < k; l++ {
						ai := i*lda + l
						if trans {
							ai = l*lda + i
						}
						a[ai] = math.Copysign(0, -1)
						if name == "mixed" && (l == 2 || l == 3) {
							a[ai] = float64(i+l) - 2.5
						} else if name == "alpha-zero" || name == "underflow" {
							a[ai] = 0.25
						}
					}
				}
				for i := range c {
					c[i] = float64(i%7) - 3
				}
				alpha := -0.75
				if name == "alpha-zero" {
					alpha = 0
				} else if name == "underflow" {
					alpha = math.SmallestNonzeroFloat64
				}
				want := slices.Clone(c)
				for i := 0; i < m; i++ {
					for l := 0; l < k; l++ {
						ai := i*lda + l
						if trans {
							ai = l*lda + i
						}
						scale := alpha * a[ai]
						if scale == 0 {
							continue
						}
						for j := 0; j < n; j++ {
							want[i*n+j] += scale * b[l*n+j]
						}
					}
				}
				aOrig, bOrig := slices.Clone(a), slices.Clone(b)
				if !dgemmSerialSIMD(trans, false, m, n, k, a, lda, b, n, c, n, alpha) {
					t.Fatal("kernel rejected full tile")
				}
				if !dgemmSIMDEqualBits(c, want) {
					t.Fatal("zero coefficient affected result")
				}
				if !dgemmSIMDEqualBits(a, aOrig) || !dgemmSIMDEqualBits(b, bOrig) {
					t.Fatal("input changed")
				}
			})
		}
	}
}

func TestDgemmSIMDNumerical(t *testing.T) {
	m, k := dgemmSIMDRows+1, 4
	n := 2*dgemmSIMDCols*simd.BroadcastFloat64s(0).Len() - 1
	tiny := math.SmallestNonzeroFloat64
	for _, test := range []struct {
		name       string
		a, b       [4]float64
		want, atol float64
	}{
		{name: "large-cancellation", a: [4]float64{math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64, -math.MaxFloat64}, b: [4]float64{1, 1, 1, 1}},
		{name: "large-finite", a: [4]float64{math.MaxFloat64 / 8, math.MaxFloat64 / 8, math.MaxFloat64 / 8, math.MaxFloat64 / 8}, b: [4]float64{1, 1, 1, 1}, want: math.MaxFloat64 / 2},
		{name: "cancellation", a: [4]float64{1 + 0x1p-27, -1, 0x1p-27, 0}, b: [4]float64{1 - 0x1p-27, 1, 0x1p-27, 0}, atol: 1e-15},
		{name: "subnormal", a: [4]float64{8 * tiny, -4 * tiny, 2 * tiny, -tiny}, b: [4]float64{0.5, 0.5, 0.5, 0.5}, want: 2 * tiny, atol: tiny},
		{name: "overflow", a: [4]float64{math.MaxFloat64}, b: [4]float64{2}, want: math.Inf(1)},
		{name: "infinity", a: [4]float64{math.Inf(-1)}, b: [4]float64{1}, want: math.Inf(-1)},
		{name: "nan", a: [4]float64{math.NaN()}, b: [4]float64{1}, want: math.NaN()},
	} {
		for _, trans := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/trans=%t", test.name, trans), func(t *testing.T) {
				lda := k
				if trans {
					lda = m
				}
				a, b, c := make([]float64, m*k), make([]float64, k*n), make([]float64, m*n)
				for i := 0; i < m; i++ {
					for l, v := range test.a {
						ai := i*lda + l
						if trans {
							ai = l*lda + i
						}
						a[ai] = v
					}
				}
				for l, v := range test.b {
					for j := 0; j < n; j++ {
						b[l*n+j] = v
					}
				}
				if !dgemmSerialSIMD(trans, false, m, n, k, a, lda, b, n, c, n, 1) {
					t.Fatal("kernel rejected full tile")
				}
				for i, got := range c {
					if !gemmSIMDClose(got, test.want, 1e-14, test.atol) {
						t.Fatalf("index=%d got=%g want=%g", i, got, test.want)
					}
				}
			})
		}
	}
}

func dgemmSIMDEqualBits(x, y []float64) bool {
	return slices.EqualFunc(x, y, func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b)
	})
}

func gemmSIMDClose(got, want, rtol, atol float64) bool {
	if math.IsNaN(want) {
		return math.IsNaN(got)
	}
	if math.IsInf(want, 0) {
		return got == want
	}
	return !math.IsNaN(got) && !math.IsInf(got, 0) && math.Abs(got-want) <= atol+rtol*math.Abs(want)
}

func BenchmarkDgemmSIMDTiles(b *testing.B) {
	for _, dims := range [][3]int{
		{8, 8, 8}, {10, 10, 10}, {16, 16, 16},
		{60, 56, 60}, {60, 59, 60}, {60, 60, 60}, {60, 61, 60}, {60, 64, 60},
		{64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {512, 64, 32}, {64, 512, 32},
	} {
		m, n, k := dims[0], dims[1], dims[2]
		for _, trans := range []bool{false, true} {
			for _, candidate := range []bool{false, true} {
				b.Run(fmt.Sprintf("%dx%dx%d/trans=%t/candidate=%t", m, n, k, trans, candidate), func(b *testing.B) {
					ar, ac := m, k
					if trans {
						ar, ac = k, m
					}
					lda, ldb, ldc := ac+3, n+3, n+5
					a, bb, c := make([]float64, ar*lda), make([]float64, k*ldb), make([]float64, m*ldc)
					rng := rand.New(rand.NewPCG(1, 2))
					for _, x := range [][]float64{a, bb, c} {
						for i := range x {
							x[i] = rng.Float64() - 0.5
						}
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if candidate && dgemmSerialSIMD(trans, false, m, n, k, a, lda, bb, ldb, c, ldc, 0.75) {
							continue
						}
						if trans {
							dgemmSerialTransNot(m, n, k, a, lda, bb, ldb, c, ldc, 0.75)
						} else {
							dgemmSerialNotNot(m, n, k, a, lda, bb, ldb, c, ldc, 0.75)
						}
					}
				})
			}
		}
	}
}
