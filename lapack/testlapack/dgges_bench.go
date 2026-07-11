// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/lapack"
)

func DggbalBenchmark(b *testing.B, impl Dggbaler) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		lscale := make([]float64, n)
		rscale := make([]float64, n)
		work := make([]float64, 6*n)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dggbal(lapack.Scale, n, a, n, bm, n, lscale, rscale, work)
			}
		})
	}
}

func DggesBenchmark(b *testing.B, impl Dggeser) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		for i := 0; i < n; i++ {
			bOrig[i*n+i] += float64(n)
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
			n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
					n, a, n, bm, n, alphar, alphai, beta,
					nil, 1, nil, 1, work, len(work), nil)
			}
		})
	}
}

func DggesScaledSortBenchmark(b *testing.B, impl Dggeser) {
	rnd := rand.New(rand.NewPCG(2, 2))
	selector := func(_, _, _ float64) bool { return false }
	for _, n := range []int{10, 50, 100, 200} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64() * 1e-200
			bOrig[i] = rnd.NormFloat64() * 1e-200
		}
		for i := 0; i < n; i++ {
			bOrig[i*n+i] += float64(n) * 1e-200
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		bwork := make([]bool, n)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
			n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector,
					n, a, n, bm, n, alphar, alphai, beta,
					nil, 1, nil, 1, work, len(work), bwork)
			}
		})
	}
}

func DggesIsolatedBenchmark(b *testing.B, impl Dggeser) {
	for _, n := range []int{10, 50, 100, 200, 500} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range n {
			aOrig[i*n+i] = float64(i + 1)
			bOrig[i*n+i] = 1
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		work := make([]float64, 1)
		impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
			n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1, nil)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
					n, a, n, bm, n, alphar, alphai, beta,
					nil, 1, nil, 1, work, len(work), nil)
			}
		})
	}
}
