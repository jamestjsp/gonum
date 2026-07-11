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
)

func TestDhgeqzNetlibDifferential(t *testing.T) {
	rnd := rand.New(rand.NewPCG(13, 13))
	for n := 2; n <= 10; n++ {
		for k := 0; k < 40; k++ {
			h := make([]float64, n*n)
			tt := make([]float64, n*n)
			for i := range n {
				for j := max(0, i-1); j < n; j++ {
					h[i*n+j] = rnd.NormFloat64()
				}
				for j := i; j < n; j++ {
					tt[i*n+j] = rnd.NormFloat64()
				}
				tt[i*n+i] += float64(n)
			}
			t.Run(fmt.Sprintf("n=%d/case=%d", n, k), func(t *testing.T) {
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
					n, 0, n-1, h, tt, true)
				compareDhgeqzWithNetlib(t, lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone,
					n, 0, n-1, h, tt, false)
			})
		}
	}
}

func TestDhgeqzNetlibFailureLeavesLeadingEigenvalues(t *testing.T) {
	const n = 4
	h := []float64{
		2, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 2, 1,
		0, 0, math.NaN(), 3,
	}
	tt := []float64{
		-1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	gh, gt := append([]float64(nil), h...), append([]float64(nil), tt...)
	nh, nt := append([]float64(nil), h...), append([]float64(nil), tt...)
	gar := []float64{11, 12, 13, 14}
	gai := []float64{21, 22, 23, 24}
	gbeta := []float64{31, 32, 33, 34}
	nar := append([]float64(nil), gar...)
	nai := append([]float64(nil), gai...)
	nbeta := append([]float64(nil), gbeta...)
	work := make([]float64, n)
	gok := Implementation{}.Dhgeqz(lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone,
		n, 1, n-1, gh, n, gt, n, gar, gai, gbeta, nil, 1, nil, 1, work, len(work))
	nq, nz := make([]float64, n*n), make([]float64, n*n)
	info := netlibDhgeqz(byte(lapack.EigenvaluesOnly), byte(lapack.SchurNone), byte(lapack.SchurNone),
		n, 1, n-1, nh, nt, nar, nai, nbeta, nq, nz)
	if gok || info <= 0 {
		t.Fatalf("failure mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if gh[0] != nh[0] || gt[0] != nt[0] || gar[0] != nar[0] || gai[0] != nai[0] || gbeta[0] != nbeta[0] {
		t.Fatalf("leading failure output mismatch: Gonum H=%g T=%g alpha=(%g,%g) beta=%g; Netlib H=%g T=%g alpha=(%g,%g) beta=%g",
			gh[0], gt[0], gar[0], gai[0], gbeta[0], nh[0], nt[0], nar[0], nai[0], nbeta[0])
	}
}

func TestDhgeqzNetlibComplexRepresentation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		h, tt []float64
	}{
		{
			name: "NonDiagonalT",
			h:    []float64{0, -1, 1, 0},
			tt:   []float64{2, 1, 0, 3},
		},
		{
			name: "SeparatedDiagonals",
			h:    []float64{4, -7, 3, 2},
			tt:   []float64{1e-100, 2e-100, 0, 4e-100},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const n = 2
			gh, gt := append([]float64(nil), tc.h...), append([]float64(nil), tc.tt...)
			nh, nt := append([]float64(nil), tc.h...), append([]float64(nil), tc.tt...)
			gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
			nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
			gq, gz := make([]float64, n*n), make([]float64, n*n)
			nq, nz := make([]float64, n*n), make([]float64, n*n)
			work := make([]float64, n)
			gok := Implementation{}.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
				n, 0, n-1, gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n, work, len(work))
			info := netlibDhgeqz(byte(lapack.EigenvaluesAndSchur), byte(lapack.SchurHess), byte(lapack.SchurHess),
				n, 0, n-1, nh, nt, nar, nai, nbeta, nq, nz)
			if !gok || info != 0 {
				t.Fatalf("Gonum success=%v Netlib info=%d", gok, info)
			}
			for i := range n {
				for _, values := range [][2]float64{{gar[i], nar[i]}, {gai[i], nai[i]}, {gbeta[i], nbeta[i]}} {
					tol := 2e-13*math.Abs(values[1]) + 1e-300
					if math.Abs(values[0]-values[1]) > tol {
						t.Fatalf("triplet representation mismatch at %d: Gonum=(%v,%v)/%v Netlib=(%v,%v)/%v",
							i, gar, gai, gbeta, nar, nai, nbeta)
					}
				}
			}
		})
	}
}

func TestDhgeqzNetlibActiveSubrange(t *testing.T) {
	const n = 6
	h := []float64{
		-2, 1, 0, 0, 0, 0,
		0, 1, 2, 0.5, 0, 0,
		0, -1, 3, 1, 0.25, 0,
		0, 0, 0.75, -1, 2, 0.5,
		0, 0, 0, -0.5, 4, 1,
		0, 0, 0, 0, 0, 7,
	}
	tt := []float64{
		2, 0, 0, 0, 0, 0,
		0, 3, 0.5, 0, 0, 0,
		0, 0, 4, 0.25, 0, 0,
		0, 0, 0, 5, 0.5, 0,
		0, 0, 0, 0, 6, 0,
		0, 0, 0, 0, 0, 8,
	}
	compareDhgeqzWithNetlib(t, lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
		n, 1, 4, h, tt, true)
}

func compareDhgeqzWithNetlib(t *testing.T, job lapack.SchurJob, compq, compz lapack.SchurComp,
	n, ilo, ihi int, h, tt []float64, checkSchur bool) {
	t.Helper()
	gh, gt := append([]float64(nil), h...), append([]float64(nil), tt...)
	nh, nt := append([]float64(nil), h...), append([]float64(nil), tt...)
	gar, gai, gbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	nar, nai, nbeta := make([]float64, n), make([]float64, n), make([]float64, n)
	gq, gz := make([]float64, n*n), make([]float64, n*n)
	nq, nz := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, n)
	gok := Implementation{}.Dhgeqz(job, compq, compz, n, ilo, ihi,
		gh, n, gt, n, gar, gai, gbeta, gq, n, gz, n, work, len(work))
	info := netlibDhgeqz(byte(job), byte(compq), byte(compz), n, ilo, ihi,
		nh, nt, nar, nai, nbeta, nq, nz)
	if gok != (info == 0) {
		t.Fatalf("success mismatch: Gonum=%v Netlib info=%d", gok, info)
	}
	if !gok {
		return
	}
	compareGeneralizedEigenvalues(t, gar, gai, gbeta, nar, nai, nbeta)
	if checkSchur {
		checkGeneralizedSchurResult(t, "Gonum DHGEQZ", h, tt, gh, gt, gq, gz, n)
		checkGeneralizedSchurResult(t, "Netlib DHGEQZ", h, tt, nh, nt, nq, nz, n)
	}
}
