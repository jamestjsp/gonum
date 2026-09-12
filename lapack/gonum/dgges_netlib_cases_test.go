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
)

var dggesPencilKinds = []string{"dense", "tiny", "singular", "near-unit", "riccati"}

// dggesComparisonPencil returns deterministic row-major pencils. All n are even.
func dggesComparisonPencil(kind string, n int) (a, b []float64) {
	a, b = make([]float64, n*n), make([]float64, n*n)
	rnd := rand.New(rand.NewPCG(19, 23))
	switch kind {
	case "dense", "tiny":
		for i := range a {
			a[i] = rnd.NormFloat64()
			b[i] = rnd.NormFloat64()
		}
		for i := range n {
			b[i*n+i] += float64(n)
		}
		if kind == "tiny" {
			for i := range a {
				a[i] *= 1e-200
				b[i] *= 1e-200
			}
		}
		return a, b
	case "singular":
		for i := range n {
			a[i*n+i] = 0.1 + 0.8*float64(i+1)/float64(n+1)
			if i%2 == 0 {
				a[i*n+i] = -a[i*n+i]
			}
			if i%3 != 0 {
				b[i*n+i] = 1
			}
			// Upper triangular couplings retain exact infinite eigenvalues.
			for j := i + 1; j < n; j++ {
				a[i*n+j] = rnd.NormFloat64() / float64(n)
				b[i*n+j] = rnd.NormFloat64() / float64(n)
			}
		}
		return a, b
	case "near-unit":
		for i := 0; i < n; i += 2 {
			radius := 1 - 1e-7
			if (i/2)%2 == 0 {
				radius = 1 + 1e-7
			}
			// Distinct angles separate subspaces despite radii near one.
			angle := 0.2 + float64(i)*2/float64(n)
			x, y := radius*math.Cos(angle), radius*math.Sin(angle)
			a[i*n+i], a[i*n+i+1], a[(i+1)*n+i], a[(i+1)*n+i+1] = x, y, -y, x
			b[i*n+i], b[(i+1)*n+i+1] = 1, 1
		}
	case "riccati":
		// H=[F,0;-I,I], J=[I,I;0,F^T] is the reduced DARE pencil
		// for Q=I and B R^-1 B^T=I. Its stable subspace has dimension n/2.
		m := n / 2
		for i := 0; i < m; i++ {
			for j := 0; j < m; j++ {
				f := 0.1 * rnd.NormFloat64() / math.Sqrt(float64(m))
				if i == j {
					f += 0.9
				}
				a[i*n+j] = f
				b[(m+j)*n+m+i] = f
			}
			a[(m+i)*n+i] = -1
			a[(m+i)*n+m+i] = 1
			b[i*n+i] = 1
			b[i*n+m+i] = 1
		}
	default:
		panic("unknown DGGES pencil")
	}
	// Apply independent orthogonal equivalences to avoid already-reduced inputs.
	for k := 0; k < 3*n; k++ {
		i, j := rnd.IntN(n), rnd.IntN(n-1)
		if j >= i {
			j++
		}
		sn, cs := math.Sincos(rnd.Float64() * 2 * math.Pi)
		for _, x := range [][]float64{a, b} {
			for col := 0; col < n; col++ {
				u, v := x[i*n+col], x[j*n+col]
				x[i*n+col], x[j*n+col] = cs*u+sn*v, cs*v-sn*u
			}
		}
		i, j = rnd.IntN(n), rnd.IntN(n-1)
		if j >= i {
			j++
		}
		sn, cs = math.Sincos(rnd.Float64() * 2 * math.Pi)
		for _, x := range [][]float64{a, b} {
			for row := 0; row < n; row++ {
				u, v := x[row*n+i], x[row*n+j]
				x[row*n+i], x[row*n+j] = cs*u+sn*v, cs*v-sn*u
			}
		}
	}
	return a, b
}

func dggesSelection(kind string, sorting bool) int {
	if !sorting {
		return 0
	}
	if kind == "near-unit" || kind == "riccati" || kind == "singular" {
		return 2
	}
	return 1
}

func dggesSelect(selection int) lapack.SchurSelect {
	if selection == 2 {
		return func(ar, ai, beta float64) bool { return math.Hypot(ar, ai) < math.Abs(beta) }
	}
	return func(ar, _, beta float64) bool { return beta != 0 && ar < 0 }
}

func dggesTranspose(a []float64, n int) []float64 {
	b := make([]float64, len(a))
	for i := range n {
		for j := range n {
			b[i*n+j] = a[j*n+i]
		}
	}
	return b
}

type dggesComparison struct {
	info                     int
	a, b, ar, ai, beta, q, z []float64
	run                      func() (int, bool)
}

func newDggesComparison(n int, origA, origB []float64, vectors string, selection int, native bool) *dggesComparison {
	c := &dggesComparison{a: make([]float64, n*n), b: make([]float64, n*n), ar: make([]float64, n), ai: make([]float64, n), beta: make([]float64, n), q: make([]float64, n*n), z: make([]float64, n*n)}
	left, right := lapack.SchurNone, lapack.SchurNone
	nl, nr := byte('N'), byte('N')
	if vectors == "both" {
		left, nl = lapack.SchurHess, 'V'
	}
	if vectors != "none" {
		right, nr = lapack.SchurHess, 'V'
	}
	if native {
		oa, ob := dggesTranspose(origA, n), dggesTranspose(origB, n)
		w := netlib.NewDggesWorkspace(n, nl, nr, selection)
		c.run = func() (int, bool) {
			copy(c.a, oa)
			copy(c.b, ob)
			sdim, info := w.Run(c.a, c.b, c.ar, c.ai, c.beta, c.q, c.z)
			c.info = info
			return sdim, info == 0
		}
	} else {
		sorting := lapack.SortNone
		if selection != 0 {
			sorting = lapack.SortSelected
		}
		selectFn := dggesSelect(selection)
		query := make([]float64, 1)
		Implementation{}.Dgges(left, right, sorting, selectFn, n, nil, n, nil, n, nil, nil, nil, nil, n, nil, n, query, -1, nil)
		work, bw := make([]float64, int(query[0])), make([]bool, n)
		c.run = func() (int, bool) {
			copy(c.a, origA)
			copy(c.b, origB)
			return Implementation{}.Dgges(left, right, sorting, selectFn, n, c.a, n, c.b, n, c.ar, c.ai, c.beta, c.q, n, c.z, n, work, len(work), bw)
		}
	}
	return c
}

func TestDggesNetlibControlPencils(t *testing.T) {
	major, minor, patch := netlib.Version()
	t.Logf("LAPACK runtime %d.%d.%d", major, minor, patch)
	for _, n := range []int{10, 50, 100, 200} {
		for _, kind := range dggesPencilKinds {
			a, b := dggesComparisonPencil(kind, n)
			for _, vectors := range []string{"none", "right", "both"} {
				for _, sorting := range []bool{false, true} {
					t.Run(fmt.Sprintf("n=%d/%s/vectors=%s/sort=%t", n, kind, vectors, sorting), func(t *testing.T) {
						selection := dggesSelection(kind, sorting)
						g := newDggesComparison(n, a, b, vectors, selection, false)
						ref := newDggesComparison(n, a, b, vectors, selection, true)
						gs, gok := g.run()
						rs, rok := ref.run()
						if !gok || !rok || gs != rs {
							t.Fatalf("Go=(%d,%t), Netlib=(%d,%t)", gs, gok, rs, rok)
						}
						if sorting && (gs == 0 || gs == n) {
							t.Fatal("fixture does not split spectrum")
						}
						if sorting && kind == "riccati" && gs != n/2 {
							t.Fatalf("stable dimension=%d, want %d", gs, n/2)
						}
						if kind == "singular" {
							// Relative eigenvalue error is undefined at infinity; use projective distance.
							compareGeneralizedEigenvaluesWithMetric(t, g.ar, g.ai, g.beta, ref.ar, ref.ai, ref.beta, generalizedChordalDistance)
							if sorting && gs != n-(n+2)/3 {
								t.Fatalf("finite eigenvalue count=%d", gs)
							}
						} else {
							compareGeneralizedEigenvalues(t, g.ar, g.ai, g.beta, ref.ar, ref.ai, ref.beta)
						}
						ref.a, ref.b = dggesTranspose(ref.a, n), dggesTranspose(ref.b, n)
						ref.q, ref.z = dggesTranspose(ref.q, n), dggesTranspose(ref.z, n)
						if sorting && vectors != "none" {
							// Compare subspace projectors, not arbitrary bases/signs.
							maxErr := 0.0
							for i := range n {
								for j := range n {
									x, y := 0.0, 0.0
									for k := 0; k < gs; k++ {
										x += g.z[i*n+k] * g.z[j*n+k]
										y += ref.z[i*n+k] * ref.z[j*n+k]
									}
									maxErr = math.Max(maxErr, math.Abs(x-y))
								}
							}
							if math.IsNaN(maxErr) || maxErr > 1e-8 {
								t.Fatalf("selected right subspace projector error=%g", maxErr)
							}
						}

						for name, c := range map[string]*dggesComparison{"Go": g, "Netlib": ref} {
							checkGeneralizedSchurStructure(t, name, c.a, c.b, n)
							if vectors != "none" {
								checkOrthogonal(t, name+" Z", c.z, n)
							}
							if vectors == "both" {
								checkGeneralizedSchurResult(t, name, a, b, c.a, c.b, c.q, c.z, n)
								t.Logf("%s relative residual A=%.3g B=%.3g", name, normalizedPencilResidual(a, c.a, c.q, c.z, n), normalizedPencilResidual(b, c.b, c.q, c.z, n))
							}
							if sorting {
								selected := dggesSelect(selection)
								for i := range n {
									if selected(c.ar[i], c.ai[i], c.beta[i]) != (i < gs) {
										t.Fatalf("%s: eigenvalue %d is not in requested partition", name, i)
									}
								}
							}
						}
					})
				}
			}
		}
	}
}

func TestPencilResidualScaleInvariant(t *testing.T) {
	q := []float64{1, 0, 0, 1}
	for _, scale := range []float64{1e-300, 1, 1e300} {
		orig := []float64{scale, 0, 0, scale}
		bad := []float64{2 * scale, 0, 0, scale}
		if got := normalizedPencilResidual(orig, bad, q, q, 2); math.Abs(got-1) > 1e-15 {
			t.Fatalf("scale=%g residual=%g, want 1", scale, got)
		}
	}
}

// Selecting exact beta!=0 at infinity is discontinuous under reordering.
// Both implementations must report the failed selection check while returning
// a valid partial Schur decomposition, rather than silently claiming success.
func TestDggesNetlibSelectionAtInfinity(t *testing.T) {
	const n = 10
	a, b := dggesComparisonPencil("singular", n)
	for i := range n {
		a[i*n+i] = float64(i + 1)
		if i%2 == 0 {
			a[i*n+i] = -a[i*n+i]
		}
	}
	g := newDggesComparison(n, a, b, "both", 1, false)
	ref := newDggesComparison(n, a, b, "both", 1, true)
	gs, gok := g.run()
	rs, rok := ref.run()
	if gok || rok || ref.info != n+2 || gs != rs {
		t.Fatalf("Go=(%d,%t), Netlib=(%d,%t,info=%d)", gs, gok, rs, rok, ref.info)
	}
	ref.a, ref.b = dggesTranspose(ref.a, n), dggesTranspose(ref.b, n)
	ref.q, ref.z = dggesTranspose(ref.q, n), dggesTranspose(ref.z, n)
	checkGeneralizedSchurResult(t, "Go failure outputs", a, b, g.a, g.b, g.q, g.z, n)
	checkGeneralizedSchurResult(t, "Netlib failure outputs", a, b, ref.a, ref.b, ref.q, ref.z, n)
}

func generalizedChordalDistance(ar, ai, beta, br, bi, gamma float64) float64 {
	scaleA := math.Max(math.Abs(ar), math.Max(math.Abs(ai), math.Abs(beta)))
	scaleB := math.Max(math.Abs(br), math.Max(math.Abs(bi), math.Abs(gamma)))
	if scaleA == 0 || scaleB == 0 {
		if scaleA == scaleB {
			return 0
		}
		return 1
	}
	ar, ai, beta = ar/scaleA, ai/scaleA, beta/scaleA
	br, bi, gamma = br/scaleB, bi/scaleB, gamma/scaleB
	return math.Hypot(ar*gamma-br*beta, ai*gamma-bi*beta) / (math.Hypot(math.Hypot(ar, ai), beta) * math.Hypot(math.Hypot(br, bi), gamma))
}
