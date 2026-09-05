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

func TestSgemmSIMDTiles(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 37))
	width := simd.BroadcastFloat32s(0).Len()
	tile := sgemmSIMDCols * width
	shapes := [][3]int{
		{sgemmSIMDRows - 1, tile, 4}, {sgemmSIMDRows, tile - 1, 4}, {sgemmSIMDRows, tile, 3},
		{sgemmSIMDRows, tile, 4}, {sgemmSIMDRows, tile + 1, 5}, {sgemmSIMDRows + 1, 2*tile - 1, 7},
		{2*sgemmSIMDRows - 1, 2*tile + 1, 17}, {2 * sgemmSIMDRows, 2 * tile, 33}, {17, 4*tile + 1, 31},
	}
	for tail := 0; tail < tile; tail++ {
		shapes = append(shapes, [3]int{sgemmSIMDRows + 1, tile + tail, 9})
	}
	for _, dims := range shapes {
		m, n, k := dims[0], dims[1], dims[2]
		for _, trans := range []bool{false, true} {
			ar, ac := m, k
			if trans {
				ar, ac = k, m
			}
			lda, ldb, ldc := ac+3, n+3, n+5
			a, b, c := make([]float32, (ar-1)*lda+ac), make([]float32, (k-1)*ldb+n), make([]float32, (m-1)*ldc+n)
			for _, x := range [][]float32{a, b, c} {
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
			}
			want := make([]float64, len(c))
			for i, v := range c {
				want[i] = float64(v)
			}
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					for l := 0; l < k; l++ {
						ai := i*lda + l
						if trans {
							ai = l*lda + i
						}
						want[i*ldc+j] += -0.75 * float64(a[ai]) * float64(b[l*ldb+j])
					}
				}
			}
			aOrig, bOrig, cOrig := slices.Clone(a), slices.Clone(b), slices.Clone(c)
			accepted := sgemmSerialSIMD(trans, false, m, n, k, a, lda, b, ldb, c, ldc, -0.75)
			if wantAccepted := m >= sgemmSIMDRows && n >= tile && k >= 4; accepted != wantAccepted {
				t.Fatalf("dims=%v trans=%t accepted=%t want=%t", dims, trans, accepted, wantAccepted)
			}
			if !sgemmSIMDEqualBits(a, aOrig) || !sgemmSIMDEqualBits(b, bOrig) {
				t.Fatal("input changed")
			}
			if !accepted {
				if !sgemmSIMDEqualBits(c, cOrig) {
					t.Fatal("rejected kernel changed C")
				}
				continue
			}
			for i, v := range want {
				if i%ldc >= n {
					if math.Float32bits(c[i]) != math.Float32bits(cOrig[i]) {
						t.Fatalf("padding changed at %d", i)
					}
				} else if !gemmSIMDClose(float64(c[i]), v, 2e-5, 2e-5) {
					t.Fatalf("dims=%v trans=%t index=%d got=%g want=%g", dims, trans, i, c[i], v)
				}
			}
		}
	}
}

func TestSgemmSIMDReject(t *testing.T) {
	m := sgemmSIMDRows + 1
	n := sgemmSIMDCols*simd.BroadcastFloat32s(0).Len() + 1
	k := n
	for _, trans := range []bool{false, true} {
		for _, alias := range []string{"transposed-b", "c=a", "c=a+1", "a=c+1", "c=b", "c=b+1", "b=c+1"} {
			t.Run(fmt.Sprintf("trans=%t/%s", trans, alias), func(t *testing.T) {
				lda := k
				if trans {
					lda = m
				}
				a, b, c := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
				shared := make([]float32, max(len(a), len(b), len(c))+1)
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
				for _, x := range [][]float32{a, b, c} {
					for i := range x {
						x[i] = float32(i%13) - 6
					}
				}
				aOrig, bOrig, cOrig := slices.Clone(a), slices.Clone(b), slices.Clone(c)
				if sgemmSerialSIMD(trans, alias == "transposed-b", m, n, k, a, lda, b, n, c, n, 0.75) {
					t.Fatal("kernel accepted unsupported layout")
				}
				if !sgemmSIMDEqualBits(a, aOrig) || !sgemmSIMDEqualBits(b, bOrig) || !sgemmSIMDEqualBits(c, cOrig) {
					t.Fatal("rejected kernel changed an operand")
				}
			})
		}
	}
}

func TestSgemmSIMDNumerical(t *testing.T) {
	m, k := sgemmSIMDRows+1, 4
	n := 2*sgemmSIMDCols*simd.BroadcastFloat32s(0).Len() - 1
	tiny := float32(math.SmallestNonzeroFloat32)
	for _, test := range []struct {
		name              string
		a, b              [4]float32
		alpha, want, atol float32
	}{
		{name: "large-cancellation", a: [4]float32{math.MaxFloat32, -math.MaxFloat32, math.MaxFloat32, -math.MaxFloat32}, b: [4]float32{1, 1, 1, 1}, alpha: 1},
		{name: "large-finite", a: [4]float32{math.MaxFloat32 / 8, math.MaxFloat32 / 8, math.MaxFloat32 / 8, math.MaxFloat32 / 8}, b: [4]float32{1, 1, 1, 1}, alpha: 1, want: math.MaxFloat32 / 2},
		{name: "cancellation", a: [4]float32{1 + 0x1p-13, -1, 0x1p-13, 0}, b: [4]float32{1 - 0x1p-13, 1, 0x1p-13, 0}, alpha: 1, atol: 1e-7},
		{name: "subnormal", a: [4]float32{8 * tiny, -4 * tiny, 2 * tiny, -tiny}, b: [4]float32{0.5, 0.5, 0.5, 0.5}, alpha: 1, want: 2 * tiny, atol: tiny},
		{name: "overflow", a: [4]float32{math.MaxFloat32}, b: [4]float32{2}, alpha: 1, want: float32(math.Inf(1))},
		{name: "infinity", a: [4]float32{float32(math.Inf(-1))}, b: [4]float32{1}, alpha: 1, want: float32(math.Inf(-1))},
		{name: "nan", a: [4]float32{float32(math.NaN())}, b: [4]float32{1}, alpha: 1, want: float32(math.NaN())},
		{name: "mixed-zero", a: [4]float32{0, 2, 0, -1}, b: [4]float32{float32(math.NaN()), 3, float32(math.Inf(1)), 4}, alpha: 1, want: 2},
		{name: "alpha-zero", a: [4]float32{1, 1, 1, 1}, b: [4]float32{float32(math.NaN()), float32(math.Inf(1)), 3, 4}},
		{name: "underflow", a: [4]float32{0.25, 0.25, 0.25, 0.25}, b: [4]float32{float32(math.NaN()), float32(math.Inf(1)), 3, 4}, alpha: tiny},
	} {
		for _, trans := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/trans=%t", test.name, trans), func(t *testing.T) {
				lda := k
				if trans {
					lda = m
				}
				a, b, c := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
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
				if !sgemmSerialSIMD(trans, false, m, n, k, a, lda, b, n, c, n, test.alpha) {
					t.Fatal("kernel rejected full tile")
				}
				for i, got := range c {
					if !gemmSIMDClose(float64(got), float64(test.want), 2e-6, float64(test.atol)) {
						t.Fatalf("index=%d got=%g want=%g", i, got, test.want)
					}
				}
			})
		}
	}
}

func sgemmSIMDEqualBits(x, y []float32) bool {
	return slices.EqualFunc(x, y, func(a, b float32) bool {
		return math.Float32bits(a) == math.Float32bits(b)
	})
}

func BenchmarkSgemmSIMDTiles(b *testing.B) {
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
					a, bb, c := make([]float32, ar*lda), make([]float32, k*ldb), make([]float32, m*ldc)
					rng := rand.New(rand.NewPCG(1, 2))
					for _, x := range [][]float32{a, bb, c} {
						for i := range x {
							x[i] = rng.Float32() - 0.5
						}
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if candidate && sgemmSerialSIMD(trans, false, m, n, k, a, lda, bb, ldb, c, ldc, 0.75) {
							continue
						}
						if trans {
							sgemmSerialTransNot(m, n, k, a, lda, bb, ldb, c, ldc, 0.75)
						} else {
							sgemmSerialNotNot(m, n, k, a, lda, bb, ldb, c, ldc, 0.75)
						}
					}
				})
			}
		}
	}
}
