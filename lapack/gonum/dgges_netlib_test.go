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

	"gonum.org/v1/gonum/lapack"
	"gonum.org/v1/gonum/lapack/gonum/internal/netlib"
	"gonum.org/v1/gonum/lapack/testlapack"
)

func TestDggesNetlibBenchmarkPencil(t *testing.T) {
	for _, n := range []int{8, 32, 64, 128, 256} {
		for _, mode := range []struct {
			name           string
			jobvsl, jobvsr lapack.SchurComp
			njobvsl        byte
			njobvsr        byte
			selection      byte
		}{
			{"NoVectorsNoSort", lapack.SchurNone, lapack.SchurNone, 'N', 'N', 'N'},
			{"VectorsNoSort", lapack.SchurHess, lapack.SchurHess, 'V', 'V', 'N'},
			{"VectorsLeftHalfPlane", lapack.SchurHess, lapack.SchurHess, 'V', 'V', 'L'},
			{"VectorsUnitDisk", lapack.SchurHess, lapack.SchurHess, 'V', 'V', 'D'},
		} {
			t.Run(fmt.Sprintf("n=%d/%s", n, mode.name), func(t *testing.T) {
				testDggesNetlibBenchmarkPencil(t, n, mode.jobvsl, mode.jobvsr, mode.njobvsl, mode.njobvsr, mode.selection)
			})
		}
	}
}

func testDggesNetlibBenchmarkPencil(t *testing.T, n int, jobvsl, jobvsr lapack.SchurComp, njobvsl, njobvsr, selection byte) {
	t.Helper()
	aOrig, bOrig := testlapack.DggesBenchmarkPencil(n)
	ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
	na := factorColMajor(n, n, aOrig, n, n)
	nb := factorColMajor(n, n, bOrig, n, n)
	gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)

	var gselect lapack.SchurSelect
	gsort := lapack.SortNone
	var gbwork []bool
	if selection != 'N' {
		gsort = lapack.SortSelected
		gselect = func(ar, ai, beta float64) bool {
			return testlapack.DggesBenchmarkSelect(selection, ar, ai, beta)
		}
		gbwork = make([]bool, n)
	}
	var gvsl, gvsr []float64
	ldvsl, ldvsr := 1, 1
	if jobvsl == lapack.SchurHess {
		gvsl, ldvsl = make([]float64, n*n), n
	}
	if jobvsr == lapack.SchurHess {
		gvsr, ldvsr = make([]float64, n*n), n
	}
	gworkq := make([]float64, 1)
	Implementation{}.Dgges(jobvsl, jobvsr, gsort, gselect, n,
		nil, n, nil, n, nil, nil, nil, nil, ldvsl, nil, ldvsr, gworkq, -1, nil)
	gwork := make([]float64, dggesWorkLen(t, "Gonum", gworkq[0]))
	gsdim, gok := Implementation{}.Dgges(jobvsl, jobvsr, gsort, gselect, n,
		ga, n, gb, n, gar, gai, gbet, gvsl, ldvsl, gvsr, ldvsr, gwork, len(gwork), gbwork)
	if !gok {
		t.Fatal("Gonum Dgges did not converge")
	}

	nvsl, nvsr := make([]float64, n*n), make([]float64, n*n)
	nworkq := make([]float64, 1)
	_, info := netlib.DggesWork(njobvsl, njobvsr, selection, n,
		na, nb, nar, nai, nbet, nvsl, nvsr, nworkq, -1, make([]int32, n))
	if info != 0 {
		t.Fatalf("Netlib workspace query info=%d", info)
	}
	nwork := make([]float64, dggesWorkLen(t, "Netlib", nworkq[0]))
	nsdim, info := netlib.DggesWork(njobvsl, njobvsr, selection, n,
		na, nb, nar, nai, nbet, nvsl, nvsr, nwork, len(nwork), make([]int32, n))
	if info != 0 {
		t.Fatalf("Netlib Dgges info=%d", info)
	}
	checkDggesFinite(t, "Gonum eigenvalues", gar, gai, gbet)
	checkDggesFinite(t, "Netlib eigenvalues", nar, nai, nbet)

	if selection == 'N' {
		if gsdim != 0 || nsdim != 0 {
			t.Fatalf("unsorted selected count: Gonum=%d Netlib=%d want 0", gsdim, nsdim)
		}
		for _, backend := range []struct {
			name         string
			ar, ai, beta []float64
		}{{"Gonum", gar, gai, gbet}, {"Netlib", nar, nai, nbet}} {
			for _, selector := range []byte{'L', 'D'} {
				if !dggesSelectionInterleaved(selector, backend.ar, backend.ai, backend.beta) {
					t.Fatalf("%s unsorted output is not interleaved for selector %c", backend.name, selector)
				}
			}
		}
	} else {
		want := n / 2
		if gsdim != want || nsdim != want {
			t.Fatalf("selected count: Gonum=%d Netlib=%d want %d", gsdim, nsdim, want)
		}
		checkDggesSelectedPrefix(t, "Gonum", selection, gsdim, gar, gai, gbet)
		checkDggesSelectedPrefix(t, "Netlib", selection, nsdim, nar, nai, nbet)
	}

	compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
	checkDggesFinite(t, "Gonum Schur form", ga, gb)
	checkGeneralizedSchurStructure(t, "Gonum benchmark pencil", ga, gb, n)
	narow := factorRowMajor(n, n, na, n, n)
	nbrow := factorRowMajor(n, n, nb, n, n)
	checkDggesFinite(t, "Netlib Schur form", narow, nbrow)
	checkGeneralizedSchurStructure(t, "Netlib benchmark pencil", narow, nbrow, n)
	if jobvsl == lapack.SchurHess && jobvsr == lapack.SchurHess {
		checkDggesResultO3(t, "Gonum benchmark pencil", aOrig, bOrig, ga, gb, gvsl, gvsr, n)
		nvslrow := factorRowMajor(n, n, nvsl, n, n)
		nvsrrow := factorRowMajor(n, n, nvsr, n, n)
		checkDggesResultO3(t, "Netlib benchmark pencil", aOrig, bOrig, narow, nbrow, nvslrow, nvsrrow, n)
	}
}

func TestDggesBenchmarkSelectEdges(t *testing.T) {
	for _, tc := range []struct {
		selection    byte
		ar, ai, beta float64
		want         bool
	}{
		{'L', -1, 0, 0, false},
		{'L', -1, 0, 2, true},
		{'L', 1, 0, -2, true},
		{'L', -1, 0, -2, false},
		{'D', 0.5, 0, 0, false},
		{'D', 0.5, 0, -1, true},
		{'D', 2, 0, -1, false},
	} {
		if got := testlapack.DggesBenchmarkSelect(tc.selection, tc.ar, tc.ai, tc.beta); got != tc.want {
			t.Errorf("selection=%c (%g,%g)/%g: got %t want %t", tc.selection, tc.ar, tc.ai, tc.beta, got, tc.want)
		}
	}
}

func dggesSelectionInterleaved(selection byte, ar, ai, beta []float64) bool {
	seenUnselected := false
	for i := range ar {
		selected := testlapack.DggesBenchmarkSelect(selection, ar[i], ai[i], beta[i])
		if selected && seenUnselected {
			return true
		}
		seenUnselected = seenUnselected || !selected
	}
	return false
}

func checkDggesSelectedPrefix(t *testing.T, name string, selection byte, sdim int, ar, ai, beta []float64) {
	t.Helper()
	for i := range ar {
		selected := testlapack.DggesBenchmarkSelect(selection, ar[i], ai[i], beta[i])
		if selected != (i < sdim) {
			t.Fatalf("%s eigenvalue %d selected=%t, want prefix length %d", name, i, selected, sdim)
		}
		if ai[i] > 0 {
			if i+1 == len(ai) || ai[i+1] >= 0 || selected != testlapack.DggesBenchmarkSelect(selection, ar[i+1], ai[i+1], beta[i+1]) {
				t.Fatalf("%s conjugate pair split or malformed at %d", name, i)
			}
		}
	}
	if sdim > 0 && ai[sdim-1] > 0 || sdim < len(ai) && ai[sdim] < 0 {
		t.Fatalf("%s selected prefix splits a conjugate pair at %d", name, sdim)
	}
}

func dggesWorkLen(t *testing.T, name string, work float64) int {
	t.Helper()
	if math.IsNaN(work) || math.IsInf(work, 0) || work < 1 || work > float64(int(^uint(0)>>1)) {
		t.Fatalf("%s invalid workspace query %g", name, work)
	}
	return int(work)
}

func checkDggesResultO3(t *testing.T, name string, aOrig, bOrig, s, tt, q, z []float64, n int) {
	t.Helper()
	checkDggesFinite(t, name, s, tt, q, z)
	checkOrthogonal(t, name+" VSL", q, n)
	checkOrthogonal(t, name+" VSR", z, n)
	checkDggesResidualO3(t, name+" A", aOrig, s, q, z, n)
	checkDggesResidualO3(t, name+" B", bOrig, tt, q, z, n)
}

func checkDggesResidualO3(t *testing.T, name string, orig, schur, q, z []float64, n int) {
	t.Helper()
	tmp := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				tmp[i*n+j] += q[i*n+k] * schur[k*n+j]
			}
		}
	}
	norm, maxErr := 0.0, 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			got := 0.0
			for k := 0; k < n; k++ {
				got += tmp[i*n+k] * z[j*n+k]
			}
			if math.IsNaN(got) || math.IsInf(got, 0) {
				t.Fatalf("%s: nonfinite reconstruction at [%d,%d]", name, i, j)
			}
			norm = math.Max(norm, math.Abs(orig[i*n+j]))
			maxErr = math.Max(maxErr, math.Abs(orig[i*n+j]-got))
		}
	}
	if maxErr > 1e-10*float64(n)*math.Max(1, norm) {
		t.Fatalf("%s: reconstruction error=%g, norm=%g", name, maxErr, norm)
	}
}

func checkDggesFinite(t *testing.T, name string, values ...[]float64) {
	t.Helper()
	for k, value := range values {
		for i, v := range value {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Fatalf("%s: nonfinite value in array %d at %d: %g", name, k, i, v)
			}
		}
	}
}

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

func TestDggesNetlibOptions(t *testing.T) {
	const n = 6
	rnd := rand.New(rand.NewPCG(7, 7))
	aOrig := make([]float64, n*n)
	bOrig := make([]float64, n*n)
	for i := range aOrig {
		aOrig[i] = rnd.NormFloat64()
		bOrig[i] = rnd.NormFloat64()
	}
	selector := func(ar, _ float64, beta float64) bool { return beta != 0 && ar < 0 }
	for _, jobs := range []struct {
		name           string
		jobvsl, jobvsr lapack.SchurComp
		njobvsl        byte
		njobvsr        byte
	}{
		{name: "NoneNone", jobvsl: lapack.SchurNone, jobvsr: lapack.SchurNone, njobvsl: 'N', njobvsr: 'N'},
		{name: "VectorsNone", jobvsl: lapack.SchurHess, jobvsr: lapack.SchurNone, njobvsl: 'V', njobvsr: 'N'},
		{name: "NoneVectors", jobvsl: lapack.SchurNone, jobvsr: lapack.SchurHess, njobvsl: 'N', njobvsr: 'V'},
		{name: "VectorsVectors", jobvsl: lapack.SchurHess, jobvsr: lapack.SchurHess, njobvsl: 'V', njobvsr: 'V'},
	} {
		for _, dosort := range []bool{false, true} {
			for _, minimum := range []bool{false, true} {
				name := fmt.Sprintf("%s/Sort=%v/MinimumWork=%v", jobs.name, dosort, minimum)
				t.Run(name, func(t *testing.T) {
					ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					gar, gai, gbet := make([]float64, n), make([]float64, n), make([]float64, n)
					nar, nai, nbet := make([]float64, n), make([]float64, n), make([]float64, n)
					var gvsl, gvsr []float64
					ldvsl, ldvsr := 1, 1
					if jobs.jobvsl == lapack.SchurHess {
						gvsl = make([]float64, n*n)
						ldvsl = n
					}
					if jobs.jobvsr == lapack.SchurHess {
						gvsr = make([]float64, n*n)
						ldvsr = n
					}
					nvsl, nvsr := make([]float64, n*n), make([]float64, n*n)
					sortMode := lapack.SortNone
					var selectFn lapack.SchurSelect
					var bwork []bool
					if dosort {
						sortMode = lapack.SortSelected
						selectFn = selector
						bwork = make([]bool, n)
					}
					workQuery := make([]float64, 1)
					Implementation{}.Dgges(jobs.jobvsl, jobs.jobvsr, sortMode, selectFn, n,
						nil, n, nil, n, nil, nil, nil, nil, ldvsl, nil, ldvsr,
						workQuery, -1, nil)
					lwork := int(workQuery[0])
					if minimum {
						lwork = max(8*n, 6*n+16)
					}
					work := make([]float64, lwork)
					gsdim, gok := Implementation{}.Dgges(jobs.jobvsl, jobs.jobvsr, sortMode, selectFn, n,
						ga, n, gb, n, gar, gai, gbet, gvsl, ldvsl, gvsr, ldvsr,
						work, lwork, bwork)
					nsdim, info := netlib.DggesOptions(jobs.njobvsl, jobs.njobvsr, n,
						na, nb, dosort, nar, nai, nbet, nvsl, nvsr)
					if gok != (info == 0) || gsdim != nsdim {
						t.Fatalf("Gonum=(sdim=%d,ok=%v), Netlib=(sdim=%d,info=%d)", gsdim, gok, nsdim, info)
					}
					if !gok {
						t.Fatal("both implementations unexpectedly failed on a finite options fixture")
					}
					compareGeneralizedEigenvalues(t, gar, gai, gbet, nar, nai, nbet)
					checkGeneralizedSchurStructure(t, "Gonum DGGES", ga, gb, n)
					checkGeneralizedSchurStructure(t, "Netlib DGGES", na, nb, n)
					if gvsl != nil {
						checkOrthogonal(t, "Gonum VSL", gvsl, n)
					}
					if gvsr != nil {
						checkOrthogonal(t, "Gonum VSR", gvsr, n)
					}
					if gvsl != nil && gvsr != nil {
						checkGeneralizedSchurResult(t, "Gonum DGGES", aOrig, bOrig, ga, gb, gvsl, gvsr, n)
						checkGeneralizedSchurResult(t, "Netlib DGGES", aOrig, bOrig, na, nb, nvsl, nvsr, n)
					}
				})
			}
		}
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
	nsdim, info := netlib.DggesLargeAlpha(n, na, nb, nar, nai, nbet, nvsl, nvsr)
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

	nsdim, info := netlib.Dgges(n, na, nb, dosort, nar, nai, nbet, nvsl, nvsr)
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
