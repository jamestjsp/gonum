// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/lapack"
)

func TestDlasrLeftVariableSIMDBoundaries(t *testing.T) {
	for _, tc := range []struct {
		m, n, padding, offset int
		pattern               string
	}{
		{2, 15, 0, 0, "dense"}, {2, 16, 0, 1, "dense"}, {2, 17, 3, 3, "dense"},
		{3, 16, 0, 0, "sparse"}, {4, 17, 3, 1, "mixed"}, {7, 31, 0, 3, "dense"},
		{16, 32, 3, 0, "sparse"}, {33, 33, 0, 1, "dense"}, {3, 64, 3, 3, "mixed"},
	} {
		for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
			t.Run(fmt.Sprintf("m=%d/n=%d/lda=%d/offset=%d/pattern=%s/direct=%c", tc.m, tc.n, tc.n+tc.padding, tc.offset, tc.pattern, direct), func(t *testing.T) {
				const guard = 4
				lda := tc.n + tc.padding
				store := dlasrSIMDData(guard + tc.offset + tc.m*lda + guard)
				a := store[guard+tc.offset : len(store)-guard]
				c, s := dlasrSIMDRotations(tc.m-1, tc.pattern)
				want := slices.Clone(store)
				dlasrLeftVariableReference(direct, tc.m, tc.n, c, s, want[guard+tc.offset:], lda)
				Implementation{}.Dlasr(blas.Left, lapack.Variable, direct, tc.m, tc.n, c, s, a, lda)
				dlasrSIMDCheck(t, store, want, 8*0x1p-52*float64(tc.m))
				for i := 0; i < tc.m; i++ {
					for j := tc.n; j < lda; j++ {
						k := guard + tc.offset + i*lda + j
						if math.Float64bits(store[k]) != math.Float64bits(want[k]) {
							t.Fatalf("padding modified at %d", k)
						}
					}
				}
			})
		}
	}
}

func TestDlasrLeftVariableSIMDIdentity(t *testing.T) {
	const m, n, lda = 4, 17, 20
	c := []float64{1, 1, 1}
	s := make([]float64, m-1)
	a := dlasrSIMDData(m * lda)
	a[0], a[1], a[2], a[3] = math.Inf(1), math.NaN(), math.Copysign(0, -1), math.SmallestNonzeroFloat64
	want := slices.Clone(a)
	for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
		got := slices.Clone(a)
		Implementation{}.Dlasr(blas.Left, lapack.Variable, direct, m, n, c, s, got, lda)
		dlasrSIMDCheckBits(t, got, want)
	}
}

func TestDlasrLeftVariableSIMDAlias(t *testing.T) {
	const m, n, lda = 4, 17, 20
	for _, alias := range []string{"c", "s"} {
		for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
			t.Run(alias+"/direct="+string(direct), func(t *testing.T) {
				got := dlasrSIMDData(m * lda)
				want := slices.Clone(got)
				gc, gs := dlasrSIMDRotations(m-1, "dense")
				wc, ws := slices.Clone(gc), slices.Clone(gs)
				if alias == "c" {
					copy(got[lda+1:], gc)
					copy(want[lda+1:], wc)
					gc, wc = got[lda+1:lda+m], want[lda+1:lda+m]
				} else {
					copy(got[lda+1:], gs)
					copy(want[lda+1:], ws)
					gs, ws = got[lda+1:lda+m], want[lda+1:lda+m]
				}
				dlasrLeftVariableReference(direct, m, n, wc, ws, want, lda)
				Implementation{}.Dlasr(blas.Left, lapack.Variable, direct, m, n, gc, gs, got, lda)
				dlasrSIMDCheck(t, got, want, 8*0x1p-52*m)
			})
		}
	}
}

func TestDlasrLeftVariableSIMDExceptional(t *testing.T) {
	for _, tc := range []struct {
		name       string
		c, s, x, y float64
		kind       string
	}{
		{name: "normalized-overflow", c: math.Sqrt(0.5), s: math.Sqrt(0.5), x: math.MaxFloat64, y: math.MaxFloat64, kind: "normalized"},
		{name: "normalized-cancellation", c: math.Sqrt(0.5), s: -math.Sqrt(0.5), x: math.MaxFloat64, y: math.MaxFloat64, kind: "normalized"},
		{name: "unnormalized-fused-classification", c: 2, s: -2, x: math.MaxFloat64, y: math.MaxFloat64, kind: "unnormalized"},
		{name: "unnormalized-overflow-cancellation", c: math.MaxFloat64, s: -math.MaxFloat64, x: 2, y: 2, kind: "unnormalized"},
		{name: "subnormal-signed-zero", c: 1, s: math.Copysign(0, -1), x: math.SmallestNonzeroFloat64, y: math.Copysign(0, -1), kind: "identity"},
		{name: "nonfinite", c: 0, s: 1, x: math.Inf(1), y: math.NaN(), kind: "nonfinite"},
	} {
		for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
			t.Run(tc.name+"/direct="+string(direct), func(t *testing.T) {
				const m, n, lda = 2, 17, 19
				got := make([]float64, m*lda)
				for j := 0; j < n; j++ {
					got[j], got[lda+j] = tc.x, tc.y
				}
				for j := n; j < lda; j++ {
					got[j], got[lda+j] = math.Copysign(0, -1), math.Copysign(0, -1)
				}
				want := slices.Clone(got)
				dlasrLeftVariableReference(direct, m, n, []float64{tc.c}, []float64{tc.s}, want, lda)
				Implementation{}.Dlasr(blas.Left, lapack.Variable, direct, m, n, []float64{tc.c}, []float64{tc.s}, got, lda)
				switch tc.kind {
				case "normalized":
					for j := 0; j < n; j++ {
						dlasrSIMDCheckNormalizedRotation(t, got[j], got[lda+j], tc.c, tc.s, tc.x, tc.y)
					}
				case "unnormalized":
					for j := 0; j < n; j++ {
						if !math.IsInf(got[j], 0) {
							t.Fatalf("column %d: upper got %g want infinity", j, got[j])
						}
						if !math.IsInf(got[lda+j], 1) {
							t.Fatalf("column %d: lower got %g want +Inf", j, got[lda+j])
						}
					}
				case "identity":
					dlasrSIMDCheckBits(t, got, want)
				case "nonfinite":
					dlasrSIMDCheck(t, got, want, 0)
				}
				for i := 0; i < m; i++ {
					for j := n; j < lda; j++ {
						k := i*lda + j
						if math.Float64bits(got[k]) != math.Float64bits(want[k]) {
							t.Fatalf("padding modified at %d", k)
						}
					}
				}
			})
		}
	}
}

func dlasrSIMDCheckNormalizedRotation(t *testing.T, gotTop, gotBottom, c, s, top, bottom float64) {
	t.Helper()
	scale := max(math.Abs(top), math.Abs(bottom))
	top /= scale
	bottom /= scale
	wantTop := s*bottom + c*top
	wantBottom := c*bottom - s*top
	limit := math.MaxFloat64 / scale
	const eps = 0x1p-52
	for _, output := range []struct {
		name      string
		got, want float64
		bound     float64
	}{
		{name: "upper", got: gotTop, want: wantTop, bound: 8 * eps * (math.Abs(s*bottom) + math.Abs(c*top))},
		{name: "lower", got: gotBottom, want: wantBottom, bound: 8 * eps * (math.Abs(c*bottom) + math.Abs(s*top))},
	} {
		if math.Abs(output.want) > limit {
			if !math.IsInf(output.got, 0) || math.Signbit(output.got) != math.Signbit(output.want) {
				t.Fatalf("%s got %g want overflow with sign %g", output.name, output.got, output.want)
			}
			continue
		}
		if math.IsNaN(output.got) || math.IsInf(output.got, 0) || math.Abs(output.got/scale-output.want) > output.bound {
			t.Fatalf("%s got %g scaled=%g want scaled=%g bound=%g", output.name, output.got, output.got/scale, output.want, output.bound)
		}
	}
}

func dlasrLeftVariableReference(direct lapack.Direct, m, n int, c, s, a []float64, lda int) {
	start, end, step := 0, m-1, 1
	if direct == lapack.Backward {
		start, end, step = m-2, -1, -1
	}
	for j := start; j != end; j += step {
		ctmp, stmp := c[j], s[j]
		if ctmp == 1 && stmp == 0 {
			continue
		}
		for i := 0; i < n; i++ {
			top, bottom := a[j*lda+i], a[(j+1)*lda+i]
			a[(j+1)*lda+i] = ctmp*bottom - stmp*top
			a[j*lda+i] = stmp*bottom + ctmp*top
		}
	}
}

func dlasrSIMDRotations(n int, pattern string) ([]float64, []float64) {
	c, s := make([]float64, n), make([]float64, n)
	for i := range c {
		theta := float64(i+1) * 0.37
		c[i], s[i] = math.Cos(theta), math.Sin(theta)
		if pattern == "sparse" && i != n/2 || pattern == "mixed" && i%2 == 0 {
			c[i], s[i] = 1, 0
		}
	}
	return c, s
}

func dlasrSIMDData(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%23-11) / 16
	}
	return x
}

func dlasrSIMDCheck(t *testing.T, got, want []float64, tol float64) {
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

func dlasrSIMDCheckBits(t *testing.T, got, want []float64) {
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
