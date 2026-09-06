// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"
)

func TestDlarfg3(t *testing.T) {
	tests := []struct {
		alpha float64
		x1    float64
		x2    float64
	}{
		{alpha: 3, x1: 4},
		{alpha: -3, x1: 4, x2: 5},
		{alpha: 1},
		{alpha: math.SmallestNonzeroFloat64, x1: -math.SmallestNonzeroFloat64, x2: math.SmallestNonzeroFloat64},
		{alpha: math.MaxFloat64 / 4, x1: -math.MaxFloat64 / 8, x2: math.MaxFloat64 / 16},
	}

	impl := Implementation{}
	for _, test := range tests {
		x := []float64{test.x1, test.x2}
		wantBeta, wantTau := impl.Dlarfg(3, test.alpha, x, 1)
		gotBeta, gotTau, gotX1, gotX2 := impl.dlarfg3(test.alpha, test.x1, test.x2)
		for name, pair := range map[string][2]float64{
			"beta": {gotBeta, wantBeta},
			"tau":  {gotTau, wantTau},
			"x1":   {gotX1, x[0]},
			"x2":   {gotX2, x[1]},
		} {
			if !closeDlarfg3(pair[0], pair[1]) {
				t.Errorf("alpha=%g x1=%g x2=%g: %s=%g, want %g", test.alpha, test.x1, test.x2, name, pair[0], pair[1])
			}
		}
	}
}

func TestDhgeqzWorkspaceQueryRequiresOutput(t *testing.T) {
	defer func() {
		if got := recover(); got != shortWork {
			t.Fatalf("panic=%v, want %q", got, shortWork)
		}
	}()
	Implementation{}.Dhgeqz('E', 'N', 'N', 0, 0, -1,
		nil, 1, nil, 1, nil, nil, nil, nil, 1, nil, 1, nil, -1)
}

func TestDoQZSweepDoubleStridedRanges(t *testing.T) {
	tests := []struct {
		n, ifirst, ilast, ifrstm, ilastm int
	}{
		{3, 0, 2, 0, 2},
		{4, 0, 2, 0, 3},
		{5, 1, 3, 0, 4},
		{8, 2, 6, 1, 7},
		{33, 1, 31, 0, 32},
	}
	for _, test := range tests {
		for _, ilq := range []bool{false, true} {
			for _, ilz := range []bool{false, true} {
				name := fmt.Sprintf("n=%d/range=%d:%d/Q=%t/Z=%t", test.n, test.ifirst, test.ilast, ilq, ilz)
				t.Run(name, func(t *testing.T) {
					h0, t0, q0, z0 := qzSweepInputs(test.n)
					hWant := append([]float64(nil), h0...)
					tWant := append([]float64(nil), t0...)
					qWant := append([]float64(nil), q0...)
					zWant := append([]float64(nil), z0...)
					Implementation{}.doQZSweepDouble(true, ilq, ilz, test.n,
						test.ifirst, test.ilast, test.ifrstm, test.ilastm,
						hWant, test.n, tWant, test.n, qWant, test.n, zWant, test.n,
						0.25, 0.125, dlamchS)

					const guard = 0x1.23456789abcdefp+200
					ldh, ldt, ldq, ldz := test.n+1, test.n+3, test.n+2, test.n+4
					h := qzSweepPadded(h0, test.n, ldh, guard)
					tt := qzSweepPadded(t0, test.n, ldt, guard)
					q := qzSweepPadded(q0, test.n, ldq, guard)
					z := qzSweepPadded(z0, test.n, ldz, guard)
					Implementation{}.doQZSweepDouble(true, ilq, ilz, test.n,
						test.ifirst, test.ilast, test.ifrstm, test.ilastm,
						h, ldh, tt, ldt, q, ldq, z, ldz,
						0.25, 0.125, dlamchS)

					checkQZSweepMatrix(t, "H", h, ldh, hWant, h0, test.n, guard,
						func(i, j int) bool {
							return i < test.ifrstm && i < test.ifirst || i > test.ilast || j < test.ifirst || j > test.ilastm
						})
					checkQZSweepMatrix(t, "T", tt, ldt, tWant, t0, test.n, guard,
						func(i, j int) bool {
							return i < test.ifrstm && i < test.ifirst || i > test.ilast || j < test.ifirst || j > test.ilastm
						})
					checkQZSweepMatrix(t, "Q", q, ldq, qWant, q0, test.n, guard,
						func(_, j int) bool { return !ilq || j < test.ifirst || j > test.ilast })
					checkQZSweepMatrix(t, "Z", z, ldz, zWant, z0, test.n, guard,
						func(_, j int) bool { return !ilz || j < test.ifirst || j > test.ilast })
				})
			}
		}
	}
}

func qzSweepInputs(n int) (h, t, q, z []float64) {
	h = make([]float64, n*n)
	t = make([]float64, n*n)
	q = make([]float64, n*n)
	z = make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j >= i-1 {
				h[i*n+j] = float64((i*11+j*7)%19-9) / 8
			}
			if j >= i {
				t[i*n+j] = float64((i*5+j*13)%17-8) / 8
			}
			q[i*n+j] = float64((i*3+j*5)%13-6) / 8
			z[i*n+j] = float64((i*7+j*3)%11-5) / 8
		}
		h[i*n+i] += float64(n + 2)
		t[i*n+i] += float64(n + 4)
	}
	return h, t, q, z
}

func qzSweepPadded(src []float64, n, stride int, guard float64) []float64 {
	dst := make([]float64, (n-1)*stride+n)
	for i := range dst {
		dst[i] = guard
	}
	for i := 0; i < n; i++ {
		copy(dst[i*stride:i*stride+n], src[i*n:(i+1)*n])
	}
	return dst
}

func checkQZSweepMatrix(t *testing.T, name string, got []float64, stride int, want, before []float64, n int, guard float64, untouched func(i, j int) bool) {
	t.Helper()
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			g, w := got[i*stride+j], want[i*n+j]
			if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
				t.Fatalf("%s[%d,%d] is non-finite: got=%v want=%v", name, i, j, g, w)
			}
			if math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("%s[%d,%d]=%v (%#x), want %v (%#x)", name, i, j, g, math.Float64bits(g), w, math.Float64bits(w))
			}
			if untouched(i, j) && math.Float64bits(g) != math.Float64bits(before[i*n+j]) {
				t.Fatalf("%s[%d,%d] changed outside live range", name, i, j)
			}
		}
		if i < n-1 {
			for j := n; j < stride; j++ {
				if got[i*stride+j] != guard {
					t.Fatalf("%s padding[%d,%d]=%v, want guard", name, i, j, got[i*stride+j])
				}
			}
		}
	}
}

func closeDlarfg3(got, want float64) bool {
	if got == want {
		return true
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	return math.Abs(got-want) <= 4*dlamchP*scale
}
