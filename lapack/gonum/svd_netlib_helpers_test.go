// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"testing"
)

func netlibColMajor(rows, cols int, row []float64, stride, ld int) []float64 {
	col := make([]float64, max(1, ld*cols))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			col[i+j*ld] = row[i*stride+j]
		}
	}
	return col
}

func netlibRowMajor(rows, cols int, col []float64, ld, stride int) []float64 {
	row := make([]float64, max(1, rows*stride))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			row[i*stride+j] = col[i+j*ld]
		}
	}
	return row
}

func netlibCheckMatrix(t *testing.T, name string, rows, cols int, got []float64, ldg int, want []float64, ldw int, tol float64) {
	t.Helper()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			g, w := got[i*ldg+j], want[i*ldw+j]
			if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
				t.Fatalf("%s[%d,%d] is non-finite: got %g, want %g", name, i, j, g, w)
			}
			if math.Abs(g-w) > tol*math.Max(1, math.Max(math.Abs(g), math.Abs(w))) {
				t.Fatalf("%s[%d,%d]=%g, want %g", name, i, j, g, w)
			}
		}
	}
}

func netlibSVDResidual(m, n int, a []float64, lda int, s, u []float64, ldu int, vt []float64, ldvt int) float64 {
	k := min(m, n)
	scale := 0.0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			scale = math.Max(scale, math.Abs(a[i*lda+j]))
		}
	}
	if scale == 0 {
		scale = 1
	}
	maxErr := 0.0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k; p++ {
				sum += u[i*ldu+p] * (s[p] / scale) * vt[p*ldvt+j]
			}
			maxErr = math.Max(maxErr, math.Abs(a[i*lda+j]/scale-sum))
		}
	}
	return maxErr
}

func netlibOrthoResidual(rows, cols int, q []float64, ldq int, columns bool) float64 {
	dim := rows
	if columns {
		dim = cols
	}
	maxErr := 0.0
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			sum := 0.0
			if columns {
				for p := 0; p < rows; p++ {
					sum += q[p*ldq+i] * q[p*ldq+j]
				}
			} else {
				for p := 0; p < cols; p++ {
					sum += q[i*ldq+p] * q[j*ldq+p]
				}
			}
			if i == j {
				sum--
			}
			maxErr = math.Max(maxErr, math.Abs(sum))
		}
	}
	return maxErr
}
