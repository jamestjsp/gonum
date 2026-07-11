// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/lapack"
)

func TestDggesNetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for n := 2; n <= 8; n++ {
		for k := 0; k < 20; k++ {
			a := make([]float64, n*n)
			b := make([]float64, n*n)
			for i := range a {
				a[i] = rnd.NormFloat64()
				b[i] = rnd.NormFloat64()
			}
			compareDggesWithNetlib(t, a, b, n, k%2 == 0)
		}
	}
	for _, tc := range []struct {
		name string
		a, b []float64
	}{
		{"real-2x2", []float64{2, 1, 1, 3}, []float64{1, 0.5, 0, 1}},
		{"complex-nondiagonal-b", []float64{0, -1, 1, 0}, []float64{2, 1, 0, 3}},
		{"singular-b", []float64{1, 2, 3, 0, 4, 5, 0, 0, 6}, []float64{0, 1, 0, 0, 2, 1, 0, 0, 0}},
		{"tiny", []float64{2e-300, -1e-300, 1e-300, 2e-300}, []float64{1e-300, 0, 0, 1e-300}},
		{"huge", []float64{2e300, -1e300, 1e300, 2e300}, []float64{1e300, 0, 0, 1e300}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n := int(math.Sqrt(float64(len(tc.a))))
			compareDggesWithNetlib(t, tc.a, tc.b, n, true)
		})
	}
}

func compareDggesWithNetlib(t *testing.T, a, b []float64, n int, dosort bool) {
	t.Helper()
	ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
	na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
	gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)
	vsl, vsr := make([]float64, n*n), make([]float64, n*n)
	workq := make([]float64, 1)
	sortMode := lapack.SortNone
	var selectFn lapack.SchurSelect
	bwork := []bool(nil)
	if dosort {
		sortMode = lapack.SortSelected
		selectFn = func(ar, ai, beta float64) bool { return beta != 0 && ar < 0 }
		bwork = make([]bool, n)
	}
	impl := Implementation{}
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, sortMode, selectFn, n,
		nil, n, nil, n, nil, nil, nil, nil, n, nil, n, workq, -1, nil)
	work := make([]float64, int(workq[0]))
	gsdim, gok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, sortMode, selectFn, n,
		ga, n, gb, n, gar, gai, gbet, vsl, n, vsr, n, work, len(work), bwork)

	nsdim, info := netlibDgges(n, na, nb, dosort, nar, nai, nbet)
	if gok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", gok, int(info))
	}
	if !gok {
		return
	}
	if gsdim != nsdim {
		t.Errorf("sdim mismatch: Gonum=%d Netlib=%d", gsdim, nsdim)
	}
	compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
}

func TestDtgex2NetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(2, 2))
	for _, blocks := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		t.Run(string(rune('0'+n1))+"x"+string(rune('0'+n2)), func(t *testing.T) {
			for k := 0; k < 100; k++ {
				n := n1 + n2
				a := make([]float64, n*n)
				b := make([]float64, n*n)
				fillSchurBlock(a, b, n, 0, n1, rnd)
				fillSchurBlock(a, b, n, n1, n2, rnd)
				for i := 0; i < n1; i++ {
					for j := n1; j < n; j++ {
						a[i*n+j] = rnd.NormFloat64()
						b[i*n+j] = rnd.NormFloat64()
					}
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				q, z := identityData(n), identityData(n)
				work := make([]float64, max(n*n, 2*n*n))
				gok := Implementation{}.Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 0, n1, n2, work, len(work))
				info := netlibDtgex2(n, na, nb, 0, n1, n2)
				if gok != (info == 0) {
					t.Fatalf("case %d: acceptance mismatch: Gonum=%v Netlib info=%d", k, gok, int(info))
				}
				if gok {
					gar, gai, gbet := schurBlockEigenvalues(ga, gb, n, 0, n2)
					nar, nai, nbet := schurBlockEigenvalues(na, nb, n, 0, n2)
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
					gar, gai, gbet = schurBlockEigenvalues(ga, gb, n, n2, n1)
					nar, nai, nbet = schurBlockEigenvalues(na, nb, n, n2, n1)
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
				}
			}
		})
	}
}

func schurBlockEigenvalues(a, b []float64, n, off, size int) (ar, ai, beta []float64) {
	if size == 1 {
		return []float64{a[off*n+off]}, []float64{0}, []float64{b[off*n+off]}
	}
	s1, s2, wr1, wr2, wi := Implementation{}.Dlag2(a[off*n+off:], n, b[off*n+off:], n)
	return []float64{wr1, wr2}, []float64{wi, -wi}, []float64{s1, s2}
}

func fillSchurBlock(a, b []float64, n, off, size int, rnd *rand.Rand) {
	if size == 1 {
		a[off*n+off] = rnd.NormFloat64()
		b[off*n+off] = rnd.Float64() + 0.5
		return
	}
	r := rnd.NormFloat64()
	x := rnd.Float64() + 0.5
	y := rnd.Float64() + 0.5
	a[off*n+off], a[off*n+off+1] = r, x
	a[(off+1)*n+off], a[(off+1)*n+off+1] = -y, r
	b[off*n+off], b[(off+1)*n+off+1] = 1, 1
}

func compareGeneralizedEigenvalues(t *testing.T, ar1, ai1, b1, ar2, ai2, b2 []float64) {
	t.Helper()
	used := make([]bool, len(ar2))
	for i := range ar1 {
		best, bestj := math.Inf(1), -1
		for j := range ar2 {
			if used[j] {
				continue
			}
			d := eigenDistance(ar1[i], ai1[i], b1[i], ar2[j], ai2[j], b2[j])
			if d < best {
				best, bestj = d, j
			}
		}
		if bestj < 0 || best > 2e-10 {
			t.Fatalf("eigenvalue %d has no Netlib match: distance=%g", i, best)
		}
		used[bestj] = true
	}
}

func eigenDistance(ar1, ai1, b1, ar2, ai2, b2 float64) float64 {
	if b1 == 0 || b2 == 0 {
		if b1 == 0 && b2 == 0 {
			return 0
		}
		return math.Inf(1)
	}
	z1 := complex(ar1/b1, ai1/b1)
	z2 := complex(ar2/b2, ai2/b2)
	return cmplxAbs(z1-z2) / math.Max(1, math.Max(cmplxAbs(z1), cmplxAbs(z2)))
}

func cmplxAbs(z complex128) float64 { return math.Hypot(real(z), imag(z)) }

func identityData(n int) []float64 {
	a := make([]float64, n*n)
	for i := 0; i < n; i++ {
		a[i*n+i] = 1
	}
	return a
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
