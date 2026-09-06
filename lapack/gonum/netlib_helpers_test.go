// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"math/rand/v2"
	"testing"
)

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

const factorPadding = 9.87654321

func factorMatrix(m, n, stride int) []float64 {
	a := make([]float64, m*stride)
	for i := 0; i < m; i++ {
		for j := n; j < stride; j++ {
			a[i*stride+j] = factorPadding
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a[i*stride+j] = math.Sin(float64(17*i+11*j+3)) + math.Cos(float64(5*i-7*j+1))
			if i == j {
				a[i*stride+j] += float64(min(m, n))
			}
		}
	}
	return a
}

func factorPivotMatrix(m, n, stride int) []float64 {
	a := factorMatrix(m, n, stride)
	for i := 0; i < m/2; i++ {
		j := m - 1 - i
		for k := 0; k < n; k++ {
			a[i*stride+k], a[j*stride+k] = a[j*stride+k], a[i*stride+k]
		}
	}
	return a
}

func factorColMajor(m, n int, row []float64, stride, ld int) []float64 {
	col := make([]float64, ld*n)
	for j := 0; j < n; j++ {
		for i := m; i < ld; i++ {
			col[i+j*ld] = factorPadding
		}
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			col[i+j*ld] = row[i*stride+j]
		}
	}
	return col
}
func factorRowMajor(m, n int, col []float64, ld, stride int) []float64 {
	row := make([]float64, m*stride)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			row[i*stride+j] = col[i+j*ld]
		}
	}
	return row
}
func factorOrthoResidual(q []float64, n int) float64 {
	r := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			s := 0.0
			for k := 0; k < n; k++ {
				s += q[k*n+i] * q[k*n+j]
			}
			if i == j {
				s--
			}
			r = math.Max(r, math.Abs(s))
		}
	}
	return r
}
func factorSPD(n, stride int) []float64 {
	a := make([]float64, n*stride)
	for i := 0; i < n; i++ {
		for j := n; j < stride; j++ {
			a[i*stride+j] = factorPadding
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			v := math.Sin(float64(13*i+7*j+1)) / float64(n)
			if i == j {
				v += float64(n)
			}
			a[i*stride+j] = v
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			v := 0.5 * (a[i*stride+j] + a[j*stride+i])
			a[i*stride+j], a[j*stride+i] = v, v
		}
	}
	return a
}

func checkFactorRowPadding(t *testing.T, a []float64, m, n, stride int) {
	t.Helper()
	for i := 0; i < m; i++ {
		for j := n; j < stride; j++ {
			if a[i*stride+j] != factorPadding {
				t.Fatalf("row padding [%d,%d] changed to %g", i, j, a[i*stride+j])
			}
		}
	}
}
func checkFactorColPadding(t *testing.T, a []float64, m, n, ld int) {
	t.Helper()
	for j := 0; j < n; j++ {
		for i := m; i < ld; i++ {
			if a[i+j*ld] != factorPadding {
				t.Fatalf("column padding [%d,%d] changed to %g", i, j, a[i+j*ld])
			}
		}
	}
}
