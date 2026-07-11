// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
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
			compareDggesWithNetlib(t, a, b, n, k%2 == 0, true)
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
			compareDggesWithNetlib(t, tc.a, tc.b, n, true, true)
		})
	}
}

func TestDggesNetlibSeparatedGlobalScales(t *testing.T) {
	rnd := rand.New(rand.NewPCG(11, 11))
	for _, exponents := range [][2]int{
		{-300, -300},
		{-300, 0},
		{0, -300},
		{300, 0},
		{0, 300},
		{300, 300},
		{-200, 200},
		{200, -200},
	} {
		scaleA := math.Pow10(exponents[0])
		scaleB := math.Pow10(exponents[1])
		t.Run(fmt.Sprintf("A=1e%d/B=1e%d", exponents[0], exponents[1]), func(t *testing.T) {
			for n := 2; n <= 8; n++ {
				for k := 0; k < 20; k++ {
					a := make([]float64, n*n)
					b := make([]float64, n*n)
					for i := range a {
						a[i] = rnd.NormFloat64() * scaleA
						b[i] = rnd.NormFloat64() * scaleB
					}
					compareDggesWithNetlib(t, a, b, n, false, true)
				}
			}
		})
	}
}

func TestDggesNetlibTwoByTwoShiftFallback(t *testing.T) {
	const n = 14
	rnd := rand.New(rand.NewPCG(93, 93))
	var a, b []float64
	for k := 0; k <= 396; k++ {
		a = make([]float64, n*n)
		b = make([]float64, n*n)
		for i := range a {
			exp := rnd.IntN(601) - 300
			a[i] = rnd.NormFloat64() * math.Pow10(exp)
			exp = rnd.IntN(601) - 300
			b[i] = rnd.NormFloat64() * math.Pow10(exp)
		}
	}
	compareDggesWithNetlib(t, a, b, n, false, false)
}

func TestDggesNetlibNearlyDefective(t *testing.T) {
	a := []float64{
		1, 1,
		1e-14, 1,
	}
	b := identityData(2)
	compareDggesWithNetlib(t, a, b, 2, false, true)
}

func TestDggesNetlibNonlocalScaleDeflation(t *testing.T) {
	a := []float64{
		1, 1, 1e300,
		1e-10, 2, 1,
		0, 1, 3,
	}
	b := identityData(3)
	compareDggesWithNetlib(t, a, b, 3, false, true)
}

func TestDggesNetlibSelectorAlphaScale(t *testing.T) {
	const n = 2
	a := []float64{1e-300, 0, 0, 1e-200}
	b := []float64{1e-300, 0, 0, 1e-200}
	ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
	na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
	gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)
	vsl, vsr := make([]float64, n*n), make([]float64, n*n)
	nvsl, nvsr := make([]float64, n*n), make([]float64, n*n)
	workq := make([]float64, 1)
	impl := Implementation{}
	selector := func(alphar, _, _ float64) bool { return alphar > 1e-299 }
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selector, n,
		nil, n, nil, n, nil, nil, nil, nil, n, nil, n, workq, -1, nil)
	work := make([]float64, int(workq[0]))
	gsdim, gok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selector, n,
		ga, n, gb, n, gar, gai, gbet, vsl, n, vsr, n, work, len(work), make([]bool, n))
	nsdim, info := netlibDggesLargeAlpha(n, na, nb, nar, nai, nbet, nvsl, nvsr)
	if gok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if gsdim != nsdim {
		t.Fatalf("sdim mismatch: Gonum=%d Netlib=%d", gsdim, nsdim)
	}
	compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
	checkGeneralizedSchurResult(t, "Gonum alpha-scale selector", a, b, ga, gb, vsl, vsr, n)
	checkGeneralizedSchurResult(t, "Netlib alpha-scale selector", a, b, na, nb, nvsl, nvsr, n)
}

func TestDtgsenNetlibWorkspaceQueries(t *testing.T) {
	for _, tc := range []struct {
		name                   string
		ijob, n, lwork, liwork int
	}{
		{name: "EmptyReorder", ijob: 0, n: 0, lwork: -1, liwork: -1},
		{name: "EmptyCondition", ijob: 1, n: 0, lwork: -1, liwork: -1},
		{name: "FloatOnly", ijob: 1, n: 2, lwork: -1, liwork: 1},
		{name: "IntegerOnly", ijob: 1, n: 2, lwork: 1, liwork: -1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			work := []float64{-1}
			iwork := []int{-1}
			var selected []bool
			var a []float64
			if tc.n != 0 {
				selected = []bool{true, false}
				a = []float64{1, 0, 0, 2}
			}
			Implementation{}.Dtgsen(tc.ijob, false, false, selected, tc.n,
				a, max(1, tc.n), nil, max(1, tc.n), nil, nil, nil,
				nil, 1, nil, 1, work, tc.lwork, iwork, tc.liwork)
			nwork, niwork, info := netlibDtgsenWorkspace(tc.ijob, tc.n, tc.lwork, tc.liwork)
			if info != 0 {
				t.Fatalf("Netlib workspace query failed with info=%d", info)
			}
			if work[0] != nwork || iwork[0] != niwork {
				t.Fatalf("Gonum workspace=(%v,%d), Netlib=(%v,%d)", work[0], iwork[0], nwork, niwork)
			}
		})
	}
}

func TestDtgsenNetlibSingularConditionEstimate(t *testing.T) {
	const n = 2
	work := make([]float64, 4*n+16)
	iwork := make([]int, n+6)
	_, _, _, _, ok := Implementation{}.Dtgsen(1, false, false, []bool{true, false}, n,
		[]float64{1, 0, 0, 1}, n, []float64{1, 0, 0, 1}, n,
		make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work), iwork, len(iwork))
	info := netlibDtgsenSingular()
	if ok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", ok, info)
	}
}

func TestDtgsenNetlibDifferential(t *testing.T) {
	const n = 5
	aOrig, bOrig, selected := dtgsenOraclePencil()

	for ijob := 0; ijob <= 5; ijob++ {
		for _, wantq := range []bool{false, true} {
			for _, wantz := range []bool{false, true} {
				name := fmt.Sprintf("ijob=%d/Q=%v/Z=%v", ijob, wantq, wantz)
				t.Run(name, func(t *testing.T) {
					ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
					nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)
					gq, gz := identityData(n), identityData(n)
					nq, nz := identityData(n), identityData(n)

					workQuery := make([]float64, 1)
					iworkQuery := make([]int, 1)
					Implementation{}.Dtgsen(ijob, wantq, wantz, selected, n,
						ga, n, gb, n, gar, gai, gbet, gq, n, gz, n,
						workQuery, -1, iworkQuery, -1)
					work := make([]float64, int(workQuery[0]))
					iwork := make([]int, iworkQuery[0])
					gm, gpl, gpr, gdif, gok := Implementation{}.Dtgsen(ijob, wantq, wantz, selected, n,
						ga, n, gb, n, gar, gai, gbet, gq, n, gz, n,
						work, len(work), iwork, len(iwork))
					nm, npl, npr, ndif, info := netlibDtgsen(ijob, wantq, wantz, selected, n,
						na, nb, nar, nai, nbet, nq, nz)
					if gok != (info == 0) || gm != nm {
						t.Fatalf("Gonum=(m=%d,ok=%v), Netlib=(m=%d,info=%d)", gm, gok, nm, info)
					}
					if !gok {
						return
					}
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
					checkGeneralizedSchurStructure(t, "Gonum DTGSEN", ga, gb, n)
					checkGeneralizedSchurStructure(t, "Netlib DTGSEN", na, nb, n)
					if wantq && wantz {
						checkGeneralizedSchurResult(t, "Gonum DTGSEN", aOrig, bOrig, ga, gb, gq, gz, n)
						checkGeneralizedSchurResult(t, "Netlib DTGSEN", aOrig, bOrig, na, nb, nq, nz, n)
					}
					if ijob == 1 || ijob >= 4 {
						checkCloseNetlib(t, "pl", gpl, npl)
						checkCloseNetlib(t, "pr", gpr, npr)
					}
					if ijob >= 2 {
						if ijob == 2 || ijob == 4 {
							checkEstimateNetlib(t, "dif[0]", gdif[0], ndif[0])
							checkEstimateNetlib(t, "dif[1]", gdif[1], ndif[1])
						} else {
							checkCloseNetlib(t, "dif[0]", gdif[0], ndif[0])
							checkCloseNetlib(t, "dif[1]", gdif[1], ndif[1])
						}
					}
				})
			}
		}
	}
}

func TestDtgsenNetlibConditionBeforeNormalization(t *testing.T) {
	const n = 4
	a := []float64{
		-3, 0.2, 0.1, 0.3,
		0, 1, 0.4, -0.2,
		0, 0, 4, 0.7,
		0, 0, 0, -2,
	}
	b := []float64{
		-1, 0.1, -0.2, 0.05,
		0, 2, 0.2, -0.1,
		0, 0, -1.5, -0.2,
		0, 0, 0, 0.8,
	}
	selected := []bool{true, true, false, false}
	for _, ijob := range []int{2, 4} {
		ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
		na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
		work := make([]float64, max(4*n+16, 2*2*(n-2)))
		iwork := make([]int, n+6)
		gm, gpl, gpr, gdif, gok := Implementation{}.Dtgsen(ijob, false, false, selected, n,
			ga, n, gb, n, make([]float64, n), make([]float64, n), make([]float64, n),
			nil, 1, nil, 1, work, len(work), iwork, len(iwork))
		nm, npl, npr, ndif, info := netlibDtgsen(ijob, false, false, selected, n,
			na, nb, make([]float64, n), make([]float64, n), make([]float64, n),
			identityData(n), identityData(n))
		if gok != (info == 0) || gm != nm {
			t.Fatalf("ijob=%d: Gonum=(m=%d,ok=%v), Netlib=(m=%d,info=%d)", ijob, gm, gok, nm, info)
		}
		checkCloseNetlib(t, "dif[0]", gdif[0], ndif[0])
		checkCloseNetlib(t, "dif[1]", gdif[1], ndif[1])
		if ijob == 4 {
			checkCloseNetlib(t, "pl", gpl, npl)
			checkCloseNetlib(t, "pr", gpr, npr)
		}
	}
}

func TestDtgsenNetlibFrobeniusEstimate(t *testing.T) {
	const n = 5
	rnd := rand.New(rand.NewPCG(11, 11))
	selected := []bool{false, false, false, true, true}
	accepted := 0
	for k := 0; k < 100; k++ {
		a := make([]float64, n*n)
		b := make([]float64, n*n)
		a[0], a[1], a[n], a[n+1] = -3, 0.5+rnd.Float64(), -0.5-rnd.Float64(), -3
		b[0], b[1], b[n+1] = 1, 0.1*rnd.NormFloat64(), 1.2
		a[2*n+2], b[2*n+2] = 4, 0.8
		a[3*n+3], a[3*n+4], a[4*n+3], a[4*n+4] = 1, 0.5+rnd.Float64(), -0.5-rnd.Float64(), 1
		b[3*n+3], b[3*n+4], b[4*n+4] = 1.5, 0.1*rnd.NormFloat64(), 1.1
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if a[i*n+j] == 0 {
					a[i*n+j] = 0.2 * rnd.NormFloat64()
				}
				if b[i*n+j] == 0 {
					b[i*n+j] = 0.2 * rnd.NormFloat64()
				}
			}
		}
		ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
		na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
		work := make([]float64, 4*n+16)
		iwork := make([]int, n+6)
		gm, _, _, gdif, gok := Implementation{}.Dtgsen(2, false, false, selected, n,
			ga, n, gb, n, make([]float64, n), make([]float64, n), make([]float64, n),
			nil, 1, nil, 1, work, len(work), iwork, len(iwork))
		nm, _, _, ndif, info := netlibDtgsen(2, false, false, selected, n,
			na, nb, make([]float64, n), make([]float64, n), make([]float64, n),
			identityData(n), identityData(n))
		if gok != (info == 0) || gm != nm {
			t.Fatalf("case %d: Gonum=(m=%d,ok=%v), Netlib=(m=%d,info=%d)", k, gm, gok, nm, info)
		}
		if !gok {
			continue
		}
		accepted++
		for i := range gdif {
			checkEstimateNetlib(t, fmt.Sprintf("case %d dif[%d]", k, i), gdif[i], ndif[i])
		}
	}
	if accepted < 90 {
		t.Fatalf("only %d of 100 oracle cases were accepted", accepted)
	}
}

func TestDtgsenNetlibReverseDifKernel(t *testing.T) {
	const n = 5
	a, b, selected := dtgsenOraclePencil()
	work := make([]float64, 4*n+16)
	iwork := make([]int, n+6)
	m, _, _, _, ok := Implementation{}.Dtgsen(0, false, false, selected, n,
		a, n, b, n, make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work), iwork, 1)
	if !ok || m != 2 {
		t.Fatalf("reorder=(m=%d,ok=%v), want (2,true)", m, ok)
	}

	m1, n1 := n-m, m
	a22, a11 := make([]float64, m1*m1), make([]float64, n1*n1)
	b22, b11 := make([]float64, m1*m1), make([]float64, n1*n1)
	copyLocalBlock(m1, a[m*n+m:], n, a22, m1)
	copyLocalBlock(n1, a, n, a11, n1)
	copyLocalBlock(m1, b[m*n+m:], n, b22, m1)
	copyLocalBlock(n1, b, n, b11, n1)
	gc, gf := make([]float64, m1*n1), make([]float64, m1*n1)
	nc, nf := make([]float64, m1*n1), make([]float64, m1*n1)
	_, gdif, gok := Implementation{}.Dtgsyl(blas.NoTrans, 3, m1, n1,
		a22, m1, a11, n1, gc, n1,
		b22, m1, b11, n1, gf, n1,
		[]float64{0}, 1, make([]int, n+6))
	_, ndif, info := netlibDtgsyl(byte(blas.NoTrans), 3, m1, n1,
		a22, a11, nc, b22, b11, nf)
	if gok != (info == 0) {
		t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
	}
	checkCloseNetlib(t, "reverse dif", gdif, ndif)
	gc, gf = make([]float64, m1*n1), make([]float64, m1*n1)
	_, gdif, gok = Implementation{}.Dtgsyl(blas.NoTrans, 3, m1, n1,
		a[m*n+m:], n, a, n, gc, n1,
		b[m*n+m:], n, b, n, gf, n1,
		[]float64{0}, 1, make([]int, n+6))
	if !gok {
		t.Fatal("strided reverse kernel reported coincident eigenvalues")
	}
	checkCloseNetlib(t, "strided reverse dif", gdif, ndif)
}

func dtgsenOraclePencil() (a, b []float64, selected []bool) {
	a = []float64{
		-3, 0.2, 0.1, 0.3, 0.4,
		0, 1, 2, 0.2, -0.1,
		0, -1, 1, 0.5, 0.25,
		0, 0, 0, 4, 0.7,
		0, 0, 0, 0, -2,
	}
	b = []float64{
		1, 0.1, -0.2, 0.05, 0.3,
		0, 2, 0, 0.2, -0.1,
		0, 0, 3, 0.15, 0.25,
		0, 0, 0, 1.5, -0.2,
		0, 0, 0, 0, 0.8,
	}
	return a, b, []bool{false, false, false, true, true}
}

func TestDtgsylNetlibBlockDifferential(t *testing.T) {
	for _, test := range []struct {
		name       string
		m, n       int
		a, b, d, e []float64
	}{
		{
			name: "1x2",
			m:    1,
			n:    2,
			a:    []float64{2},
			b:    []float64{1, 2, -1, 1},
			d:    []float64{3},
			e:    []float64{2, 0.2, 0, 3},
		},
		{
			name: "2x1",
			m:    2,
			n:    1,
			a:    []float64{1, 2, -1, 1},
			b:    []float64{4},
			d:    []float64{2, 0.2, 0, 3},
			e:    []float64{1.5},
		},
		{
			name: "2x2",
			m:    2,
			n:    2,
			a:    []float64{1, 2, -1, 1},
			b:    []float64{3, 1, -0.5, 3},
			d:    []float64{2, 0.2, 0, 3},
			e:    []float64{1.5, -0.1, 0, 2.5},
		},
		{
			name: "3x2-multiblock",
			m:    3,
			n:    2,
			a: []float64{
				-3, 0.1, 0.2,
				0, 1, 2,
				0, -1, 1,
			},
			b: []float64{4, 0.7, 0, -2},
			d: []float64{
				1, 0.1, -0.2,
				0, 2, 0,
				0, 0, 3,
			},
			e: []float64{1.5, -0.2, 0, 0.8},
		},
	} {
		for _, ijob := range []int{3, 4} {
			t.Run(fmt.Sprintf("%s/ijob=%d", test.name, ijob), func(t *testing.T) {
				gc := make([]float64, test.m*test.n)
				gf := make([]float64, test.m*test.n)
				nc := make([]float64, test.m*test.n)
				nf := make([]float64, test.m*test.n)
				work := make([]float64, 1)
				gscale, gdif, gok := Implementation{}.Dtgsyl(blas.NoTrans, ijob, test.m, test.n,
					test.a, test.m, test.b, test.n, gc, test.n,
					test.d, test.m, test.e, test.n, gf, test.n,
					work, len(work), make([]int, test.m+test.n+6))
				nscale, ndif, info := netlibDtgsyl(byte(blas.NoTrans), ijob, test.m, test.n,
					test.a, test.b, nc, test.d, test.e, nf)
				if gok != (info == 0) {
					t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
				}
				checkCloseNetlib(t, "scale", gscale, nscale)
				checkCloseNetlib(t, "dif", gdif, ndif)
			})
		}
	}
}

func TestDtgsy2NetlibSuccessiveBlocks(t *testing.T) {
	const n = 3
	a := []float64{
		2, 0.1, 0.2,
		0, 1, 2,
		0, -1, 1,
	}
	b := []float64{
		4, 0.2, 0.3,
		0, 3, 1,
		0, -0.5, 3,
	}
	d := []float64{
		1, 0.1, -0.2,
		0, 2, 0.2,
		0, 0, 3,
	}
	e := []float64{
		1.5, -0.1, 0.2,
		0, 2.5, 0.3,
		0, 0, 0.7,
	}
	c := []float64{1, -2, 0.5, 0.7, 1.2, -0.3, -0.8, 0.4, 2}
	f := []float64{-0.5, 1, 0.2, 1.5, -0.7, 0.9, 0.3, -1.1, 0.6}
	for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
		gc, gf := append([]float64(nil), c...), append([]float64(nil), f...)
		nc, nf := append([]float64(nil), c...), append([]float64(nil), f...)
		gscale, gsum, gscal, gpq, gok := Implementation{}.Dtgsy2(trans, 0, n, n,
			a, n, b, n, gc, n, d, n, e, n, gf, n, 1, 0, make([]int, 2*n+2))
		nscale, nsum, nscal, npq, info := netlibDtgsy2(byte(trans), 0, n, n,
			a, b, nc, d, e, nf, 1, 0)
		if gok != (info == 0) || gpq != npq {
			t.Fatalf("trans=%v: Gonum=(pq=%d,ok=%v), Netlib=(pq=%d,info=%d)", trans, gpq, gok, npq, info)
		}
		checkCloseNetlib(t, "scale", gscale, nscale)
		checkCloseNetlib(t, "rdsum", gsum, nsum)
		checkCloseNetlib(t, "rdscal", gscal, nscal)
		for i := range gc {
			checkCloseNetlib(t, fmt.Sprintf("trans=%v C[%d]", trans, i), gc[i], nc[i])
			checkCloseNetlib(t, fmt.Sprintf("trans=%v F[%d]", trans, i), gf[i], nf[i])
		}
	}
}

func TestDlatdfNetlibEightByEight(t *testing.T) {
	z := []float64{
		1, 2, 0, 0, -3, 0, 0.5, 0,
		-1, 1, 0, 0, 0, -3, 0, 0.5,
		0, 0, 1, 2, -1, 0, -3, 0,
		0, 0, -1, 1, 0, -1, 0, -3,
		2, 0.2, 0, 0, -1.5, 0, 0.1, 0,
		0, 3, 0, 0, 0, -1.5, 0, 0.1,
		0, 0, 2, 0.2, 0, 0, -2.5, 0,
		0, 0, 0, 3, 0, 0, 0, -2.5,
	}
	gz := append([]float64(nil), z...)
	nz := append([]float64(nil), z...)
	grhs := make([]float64, 8)
	nrhs := make([]float64, 8)
	ipiv := make([]int, 8)
	jpiv := make([]int, 8)
	Implementation{}.Dgetc2(8, gz, 8, ipiv, jpiv)
	gscale, gsum := Implementation{}.Dlatdf(lapack.LocalLookAhead, 8, gz, 8,
		grhs, 1, 0, ipiv, jpiv)
	nsum, nscale, _ := netlibDlatdf(lapack.LocalLookAhead, 8, nz, nrhs, 1, 0)
	checkCloseNetlib(t, "scale", gscale, nscale)
	checkCloseNetlib(t, "sum", gsum, nsum)
	for i := range grhs {
		checkCloseNetlib(t, fmt.Sprintf("rhs[%d]", i), grhs[i], nrhs[i])
	}
}

func checkCloseNetlib(t *testing.T, name string, got, want float64) {
	t.Helper()
	tol := 5e-11 * math.Max(1, math.Abs(want))
	if math.Abs(got-want) > tol {
		t.Errorf("%s=%g, want Netlib %g (tolerance %g)", name, got, want, tol)
	}
}

func checkEstimateNetlib(t *testing.T, name string, got, want float64) {
	t.Helper()
	tol := 0.1 * math.Max(math.Abs(want), math.SmallestNonzeroFloat64)
	if math.Abs(got-want) > tol {
		t.Errorf("%s=%g, want Netlib estimate %g (tolerance %g)", name, got, want, tol)
	}
}

func compareDggesWithNetlib(t *testing.T, a, b []float64, n int, dosort, compareEigenvalues bool) {
	t.Helper()
	ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
	na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
	gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)
	vsl, vsr := make([]float64, n*n), make([]float64, n*n)
	nvsl, nvsr := make([]float64, n*n), make([]float64, n*n)
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

	nsdim, info := netlibDgges(n, na, nb, dosort, nar, nai, nbet, nvsl, nvsr)
	if gok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", gok, int(info))
	}
	if !gok {
		t.Fatal("oracle case did not converge")
	}
	if gsdim != nsdim {
		t.Errorf("sdim mismatch: Gonum=%d Netlib=%d", gsdim, nsdim)
	}
	if compareEigenvalues {
		compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
	}
	checkGeneralizedSchurResult(t, fmt.Sprintf("Gonum sort=%v", dosort), a, b, ga, gb, vsl, vsr, n)
	checkGeneralizedSchurResult(t, fmt.Sprintf("Netlib sort=%v", dosort), a, b, na, nb, nvsl, nvsr, n)
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

func TestDtgex2NetlibRejectedSwap(t *testing.T) {
	const n = 4
	a := []float64{
		-2.1908605110042063, 0.5720105957485505, -0.12695320692010448, 0.11807691769627293,
		-1.0523887992314855, -2.1908605110042063, -0.060129294742167196, -0.0023051154401202423,
		0, 0, -2.1908605110042063, 0.5720105957485505,
		0, 0, -1.0523887992314855, -2.1908605110042063,
	}
	b := []float64{
		1, 0, 0.004174493657635736, -0.14712312362881694,
		0, 1, 0.050958523233562585, 0.017831076666835567,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
	na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
	q, z := identityData(n), identityData(n)
	work := make([]float64, 2*n*n)
	gok := (Implementation{}).Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 0, 2, 2, work, len(work))
	info := netlibDtgex2(n, na, nb, 0, 2, 2)
	if gok != (info == 0) {
		t.Fatalf("acceptance mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if gok {
		t.Fatal("rejection fixture was accepted")
	}
	identity := identityData(n)
	for i := range a {
		if ga[i] != a[i] || gb[i] != b[i] || na[i] != a[i] || nb[i] != b[i] || q[i] != identity[i] || z[i] != identity[i] {
			t.Fatalf("rejected swap modified outputs at %d", i)
		}
	}
}

func TestDtgex2NetlibSeparatedScales(t *testing.T) {
	rnd := rand.New(rand.NewPCG(3, 3))
	for _, blocks := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		for _, scales := range [][2]float64{{1e-300, 1}, {1, 1e-300}, {1e300, 1}, {1, 1e300}} {
			for k := 0; k < 2000; k++ {
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
				for i := range a {
					a[i] *= scales[0]
					b[i] *= scales[1]
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				q, z := identityData(n), identityData(n)
				work := make([]float64, max(n*n, 2*n*n))
				gok := Implementation{}.Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 0, n1, n2, work, len(work))
				info := netlibDtgex2(n, na, nb, 0, n1, n2)
				if gok != (info == 0) {
					t.Fatalf("blocks=%dx%d scales=(%g,%g) case=%d: acceptance mismatch: Gonum=%v Netlib info=%d",
						n1, n2, scales[0], scales[1], k, gok, info)
				}
			}
		}
	}
}

func TestDtgex2NetlibEmbeddedBlocks(t *testing.T) {
	rnd := rand.New(rand.NewPCG(5, 5))
	for _, blocks := range [][2]int{{1, 1}, {1, 2}, {2, 1}, {2, 2}} {
		n1, n2 := blocks[0], blocks[1]
		t.Run(fmt.Sprintf("%dx%d", n1, n2), func(t *testing.T) {
			for k := 0; k < 50; k++ {
				n := n1 + n2 + 2
				a, b := make([]float64, n*n), make([]float64, n*n)
				fillSchurBlock(a, b, n, 0, 1, rnd)
				fillSchurBlock(a, b, n, 1, n1, rnd)
				fillSchurBlock(a, b, n, 1+n1, n2, rnd)
				fillSchurBlock(a, b, n, n-1, 1, rnd)
				for i := 0; i < n; i++ {
					for j := i + 1; j < n; j++ {
						inFirst := i >= 1 && j < 1+n1
						inSecond := i >= 1+n1 && j < 1+n1+n2
						if inFirst || inSecond {
							continue
						}
						a[i*n+j] = rnd.NormFloat64()
						b[i*n+j] = rnd.NormFloat64()
					}
				}
				ga, gb := append([]float64(nil), a...), append([]float64(nil), b...)
				na, nb := append([]float64(nil), a...), append([]float64(nil), b...)
				q, z := identityData(n), identityData(n)
				work := make([]float64, max(n*(n1+n2), 2*(n1+n2)*(n1+n2)))
				gok := Implementation{}.Dtgex2(true, true, n, ga, n, gb, n, q, n, z, n, 1, n1, n2, work, len(work))
				info := netlibDtgex2(n, na, nb, 1, n1, n2)
				if gok != (info == 0) {
					t.Fatalf("case %d: acceptance mismatch: Gonum=%v Netlib info=%d", k, gok, info)
				}
				if gok {
					checkGeneralizedSchurResult(t, "Gonum embedded", a, b, ga, gb, q, z, n)
					checkGeneralizedSchurStructure(t, "Netlib embedded", na, nb, n)
				}
			}
		})
	}
}

func TestDtgexcNetlibSplitBlock(t *testing.T) {
	const n = 8
	aOrig := []float64{
		1, 1, 1.1, 1.3, 2, 3, -4.7, 3.3,
		1, 1, 3.7, 7.9, 4, 5.3, 3.3, -0.9,
		0, 0, 2, -3, 3.4, 6.5, 5.2, 1.8,
		0, 0, 4, 2, -5.3, -8.9, -0.2, -0.5,
		0, 0, 0, 0, 4.2, 2, 3.3, 2.3,
		0, 0, 0, 0, 3.7, 4.2, 9.9, 8.8,
		0, 0, 0, 0, 0, 0, 9.9, 8.8,
		0, 0, 0, 0, 0, 0, -9.9, 9.9,
	}
	for _, move := range []struct {
		name       string
		ifst, ilst int
	}{{"Down", 0, 4}, {"Up", 4, 0}} {
		t.Run(move.name, func(t *testing.T) {
			bOrig := identityData(n)
			ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
			na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
			q, z := identityData(n), identityData(n)
			work := make([]float64, 4*n+16)
			gifst, gilst, gok := Implementation{}.Dtgexc(true, true, n, ga, n, gb, n, q, n, z, n, move.ifst, move.ilst, work, len(work))
			nifst, nilst, info := netlibDtgexc(n, na, nb, move.ifst, move.ilst)
			if gok != (info == 0) || gifst != nifst || gilst != nilst {
				t.Fatalf("Gonum=(%d,%d,%v), Netlib=(%d,%d,info=%d)", gifst, gilst, gok, nifst, nilst, info)
			}
			for i := 0; i < n-1; i++ {
				if (ga[(i+1)*n+i] == 0) != (na[(i+1)*n+i] == 0) {
					t.Fatalf("block split differs at %d: Gonum subdiagonal=%g, Netlib subdiagonal=%g", i, ga[(i+1)*n+i], na[(i+1)*n+i])
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
			t.Fatalf("eigenvalue %d has no Netlib match: distance=%g; Gonum=(%v,%v)/%v Netlib=(%v,%v)/%v",
				i, best, ar1, ai1, b1, ar2, ai2, b2)
		}
		used[bestj] = true
	}
}

func checkGeneralizedSchurResult(t *testing.T, name string, aOrig, bOrig, s, tt, q, z []float64, n int) {
	t.Helper()
	checkGeneralizedSchurStructure(t, name, s, tt, n)
	checkOrthogonal(t, name+" VSL", q, n)
	checkOrthogonal(t, name+" VSR", z, n)
	checkPencilResidual(t, name+" A", aOrig, s, q, z, n)
	checkPencilResidual(t, name+" B", bOrig, tt, q, z, n)
}

func checkGeneralizedSchurStructure(t *testing.T, name string, s, tt []float64, n int) {
	t.Helper()
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			if tt[i*n+j] != 0 {
				t.Fatalf("%s: T[%d,%d]=%g, want exact zero", name, i, j, tt[i*n+j])
			}
			if j < i-1 && s[i*n+j] != 0 {
				t.Fatalf("%s: S[%d,%d]=%g below first subdiagonal", name, i, j, s[i*n+j])
			}
		}
	}
	for i := 0; i < n-1; i++ {
		if s[(i+1)*n+i] == 0 {
			continue
		}
		if i > 0 && s[i*n+i-1] != 0 || i+2 < n && s[(i+2)*n+i+1] != 0 {
			t.Fatalf("%s: overlapping 2x2 block at %d", name, i)
		}
		if tt[i*n+i+1] != 0 || tt[i*n+i] == 0 || tt[(i+1)*n+i+1] == 0 {
			t.Fatalf("%s: non-canonical T block at %d", name, i)
		}
		a11, a12 := s[i*n+i], s[i*n+i+1]
		a21, a22 := s[(i+1)*n+i], s[(i+1)*n+i+1]
		b11, b22 := tt[i*n+i], tt[(i+1)*n+i+1]
		as := math.Max(math.Abs(a11), math.Max(math.Abs(a12), math.Max(math.Abs(a21), math.Abs(a22))))
		bs := math.Max(math.Abs(b11), math.Abs(b22))
		d := a11/as*(b22/bs) - a22/as*(b11/bs)
		disc := d*d + 4*(b11/bs)*(b22/bs)*(a12/as)*(a21/as)
		if disc >= 0 {
			t.Fatalf("%s: real pair left in 2x2 block at %d", name, i)
		}
		i++
	}
}

func checkOrthogonal(t *testing.T, name string, q []float64, n int) {
	t.Helper()
	maxErr := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dot := 0.0
			for k := 0; k < n; k++ {
				dot += q[k*n+i] * q[k*n+j]
			}
			want := 0.0
			if i == j {
				want = 1
			}
			maxErr = math.Max(maxErr, math.Abs(dot-want))
		}
	}
	if maxErr > 1e-11*float64(n) {
		t.Fatalf("%s: orthogonality error=%g", name, maxErr)
	}
}

func checkPencilResidual(t *testing.T, name string, orig, schur, q, z []float64, n int) {
	t.Helper()
	norm, maxErr := 0.0, 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			got := 0.0
			for k := 0; k < n; k++ {
				for l := 0; l < n; l++ {
					got += q[i*n+k] * schur[k*n+l] * z[j*n+l]
				}
			}
			norm = math.Max(norm, math.Abs(orig[i*n+j]))
			maxErr = math.Max(maxErr, math.Abs(orig[i*n+j]-got))
		}
	}
	if maxErr > 1e-10*float64(n)*math.Max(1, norm) {
		t.Fatalf("%s: reconstruction error=%g, norm=%g", name, maxErr, norm)
	}
}

func eigenDistance(ar1, ai1, b1, ar2, ai2, b2 float64) float64 {
	scale1 := math.Max(math.Abs(ar1), math.Max(math.Abs(ai1), math.Abs(b1)))
	scale2 := math.Max(math.Abs(ar2), math.Max(math.Abs(ai2), math.Abs(b2)))
	if scale1 == 0 || scale2 == 0 {
		if scale1 == scale2 {
			return 0
		}
		return 1
	}
	a1 := complex(ar1/scale1, ai1/scale1)
	a2 := complex(ar2/scale2, ai2/scale2)
	b1 /= scale1
	b2 /= scale2
	num := cmplxAbs(a1*complex(b2, 0) - a2*complex(b1, 0))
	den := cmplxAbs(a1)*math.Abs(b2) + cmplxAbs(a2)*math.Abs(b1)
	if den == 0 {
		return num
	}
	return num / den
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
