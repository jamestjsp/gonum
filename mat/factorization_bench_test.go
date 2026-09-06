// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

var factorizationBenchResult float64

func BenchmarkFactorization(b *testing.B) {
	for _, n := range []int{32, 128, 256} {
		for _, shape := range []struct {
			name string
			m, n int
		}{
			{name: "square", m: n, n: n},
			{name: "tall", m: 2 * n, n: n},
			{name: "wide", m: n, n: 2 * n},
		} {
			for _, kind := range []struct {
				name string
				kind SVDKind
			}{
				{name: "none", kind: SVDNone},
				{name: "thin", kind: SVDThin},
			} {
				b.Run(fmt.Sprintf("SVD/kind=%s/shape=%s/n=%d", kind.name, shape.name, n), func(b *testing.B) {
					a := factorizationBenchGeneral(shape.m, shape.n)
					var svd SVD
					if !svd.Factorize(a, kind.kind) {
						b.Fatal("SVD did not converge")
					}
					b.ReportAllocs()
					b.ResetTimer()
					for range b.N {
						if !svd.Factorize(a, kind.kind) {
							b.Fatal("SVD did not converge")
						}
					}
					b.StopTimer()
					factorizationBenchResult = svd.Values(nil)[0]
				})
			}
		}

		for _, shape := range []struct {
			name string
			m, n int
		}{
			{name: "square", m: n, n: n},
			{name: "tall", m: 2 * n, n: n},
		} {
			b.Run(fmt.Sprintf("QR/shape=%s/n=%d", shape.name, n), func(b *testing.B) {
				a := factorizationBenchGeneral(shape.m, shape.n)
				var qr QR
				qr.Factorize(a)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					qr.Factorize(a)
				}
				b.StopTimer()
				factorizationBenchResult = qr.Cond()
			})
		}

		b.Run(fmt.Sprintf("LQ/shape=wide/n=%d", n), func(b *testing.B) {
			a := factorizationBenchGeneral(n, 2*n)
			var lq LQ
			lq.Factorize(a)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				lq.Factorize(a)
			}
			b.StopTimer()
			factorizationBenchResult = lq.Cond()
		})

		b.Run(fmt.Sprintf("LU/n=%d", n), func(b *testing.B) {
			a := factorizationBenchGeneral(n, n)
			var lu LU
			lu.Factorize(a)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				lu.Factorize(a)
			}
			b.StopTimer()
			factorizationBenchResult = lu.Cond()
		})

		b.Run(fmt.Sprintf("Cholesky/n=%d", n), func(b *testing.B) {
			a := factorizationBenchSymmetric(n)
			var chol Cholesky
			if !chol.Factorize(a) {
				b.Fatal("Cholesky factorization failed")
			}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if !chol.Factorize(a) {
					b.Fatal("Cholesky factorization failed")
				}
			}
			b.StopTimer()
			factorizationBenchResult = chol.Cond()
		})

		for _, vectors := range []bool{false, true} {
			b.Run(fmt.Sprintf("EigenSym/vectors=%t/n=%d", vectors, n), func(b *testing.B) {
				a := factorizationBenchSymmetric(n)
				var eig EigenSym
				if !eig.Factorize(a, vectors) {
					b.Fatal("symmetric eigendecomposition failed")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if !eig.Factorize(a, vectors) {
						b.Fatal("symmetric eigendecomposition failed")
					}
				}
				b.StopTimer()
				factorizationBenchResult = eig.Values(nil)[0]
			})
		}

		for _, kind := range []struct {
			name string
			kind EigenKind
		}{
			{name: "none", kind: EigenNone},
			{name: "right", kind: EigenRight},
		} {
			b.Run(fmt.Sprintf("Eigen/kind=%s/n=%d", kind.name, n), func(b *testing.B) {
				a := factorizationBenchGeneral(n, n)
				var eig Eigen
				if !eig.Factorize(a, kind.kind) {
					b.Fatal("eigendecomposition failed")
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if !eig.Factorize(a, kind.kind) {
						b.Fatal("eigendecomposition failed")
					}
				}
				b.StopTimer()
				factorizationBenchResult = real(eig.Values(nil)[0])
			})
		}
	}
}

func BenchmarkFactorizationSolve(b *testing.B) {
	// Factorizations and right-hand sides are prepared before timing. The
	// destination is reused, so allocation counts reflect steady-state solves.
	for _, n := range []int{32, 128, 256} {
		for _, nrhs := range []int{1, 16} {
			name := fmt.Sprintf("n=%d/nrhs=%d", n, nrhs)

			b.Run("QR/"+name, func(b *testing.B) {
				a := factorizationBenchGeneral(2*n, n)
				rhs := factorizationBenchGeneral(2*n, nrhs)
				var qr QR
				qr.Factorize(a)
				var dst Dense
				if err := qr.SolveTo(&dst, false, rhs); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if err := qr.SolveTo(&dst, false, rhs); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				factorizationBenchResult = dst.At(0, 0)
			})

			b.Run("LQ/"+name, func(b *testing.B) {
				a := factorizationBenchGeneral(n, 2*n)
				rhs := factorizationBenchGeneral(n, nrhs)
				var lq LQ
				lq.Factorize(a)
				var dst Dense
				if err := lq.SolveTo(&dst, false, rhs); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if err := lq.SolveTo(&dst, false, rhs); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				factorizationBenchResult = dst.At(0, 0)
			})

			b.Run("LU/"+name, func(b *testing.B) {
				a := factorizationBenchGeneral(n, n)
				rhs := factorizationBenchGeneral(n, nrhs)
				var lu LU
				lu.Factorize(a)
				var dst Dense
				if err := lu.SolveTo(&dst, false, rhs); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if err := lu.SolveTo(&dst, false, rhs); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				factorizationBenchResult = dst.At(0, 0)
			})

			b.Run("Cholesky/"+name, func(b *testing.B) {
				a := factorizationBenchSymmetric(n)
				rhs := factorizationBenchGeneral(n, nrhs)
				var chol Cholesky
				if !chol.Factorize(a) {
					b.Fatal("Cholesky factorization failed")
				}
				var dst Dense
				if err := chol.SolveTo(&dst, rhs); err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					if err := chol.SolveTo(&dst, rhs); err != nil {
						b.Fatal(err)
					}
				}
				b.StopTimer()
				factorizationBenchResult = dst.At(0, 0)
			})

			b.Run("SVD/"+name, func(b *testing.B) {
				a := factorizationBenchGeneral(2*n, n)
				rhs := factorizationBenchGeneral(2*n, nrhs)
				var svd SVD
				if !svd.Factorize(a, SVDThin) {
					b.Fatal("SVD did not converge")
				}
				var dst Dense
				svd.SolveTo(&dst, rhs, n)
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					residuals := svd.SolveTo(&dst, rhs, n)
					factorizationBenchResult = residuals[0]
				}
				b.StopTimer()
			})
		}
	}
}

func factorizationBenchGeneral(rows, cols int) *Dense {
	rnd := rand.New(rand.NewPCG(uint64(rows), uint64(cols)))
	data := make([]float64, rows*cols)
	scale := 1 / float64(max(rows, cols))
	for i := range rows {
		for j := range cols {
			v := rnd.NormFloat64() * scale
			if i == j {
				v += 2 + float64(i)/float64(min(rows, cols))
			}
			data[i*cols+j] = v
		}
	}
	return NewDense(rows, cols, data)
}

func factorizationBenchSymmetric(n int) *SymDense {
	rnd := rand.New(rand.NewPCG(uint64(n), uint64(n)))
	a := NewSymDense(n, nil)
	for i := range n {
		a.SetSym(i, i, 4+float64(i)/float64(n))
		for j := i + 1; j < n; j++ {
			a.SetSym(i, j, 0.1*rnd.NormFloat64()/float64(n))
		}
	}
	return a
}
