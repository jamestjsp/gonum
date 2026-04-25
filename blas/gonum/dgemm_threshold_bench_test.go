// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
)

// J10 calibration benches: square M=N=K at {64,128,256,512,1024}.
// Compares dgemmSerial vs dgemmParallel directly to find the smallest
// size where parallel beats serial. Run with:
//
//	go test ./blas/gonum/... -run=NONE -bench=BenchmarkDgemmJ10 -benchtime=2x
func BenchmarkDgemmJ10Serial(b *testing.B) {
	for _, sz := range []int{64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("n=%d", sz), func(b *testing.B) {
			benchDgemmJ10(b, sz, false)
		})
	}
}

func BenchmarkDgemmJ10Parallel(b *testing.B) {
	for _, sz := range []int{64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("n=%d", sz), func(b *testing.B) {
			benchDgemmJ10(b, sz, true)
		})
	}
}

func benchDgemmJ10(b *testing.B, n int, parallel bool) {
	rnd := rand.New(rand.NewPCG(uint64(n), 17))
	a := make([]float64, n*n)
	bb := make([]float64, n*n)
	c := make([]float64, n*n)
	for i := range a {
		a[i] = rnd.NormFloat64()
		bb[i] = rnd.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if parallel {
			dgemmParallel(false, false, n, n, n, a, n, bb, n, c, n, 1)
		} else {
			dgemmSerial(false, false, n, n, n, a, n, bb, n, c, n, 1)
		}
	}
}

// J10 large-K shapes: small M=N, large K. With the legacy threshold
// (blocks(m)*blocks(n) >= 4) these stay serial regardless of K. The new
// FLOP-based threshold (M*N*K) lets them go parallel.
func BenchmarkDgemmJ10LargeKSerial(b *testing.B) {
	for _, mn := range []int{32, 64, 96} {
		for _, k := range []int{512, 1024, 4096} {
			b.Run(fmt.Sprintf("mn=%d_k=%d", mn, k), func(b *testing.B) {
				benchDgemmJ10Rect(b, mn, mn, k, false)
			})
		}
	}
}

func BenchmarkDgemmJ10LargeKParallel(b *testing.B) {
	for _, mn := range []int{32, 64, 96} {
		for _, k := range []int{512, 1024, 4096} {
			b.Run(fmt.Sprintf("mn=%d_k=%d", mn, k), func(b *testing.B) {
				benchDgemmJ10Rect(b, mn, mn, k, true)
			})
		}
	}
}

// J10 end-to-end Dgemm bench (post-threshold dispatch). This is the
// number that matters for the AC: Dgemm at n=1024 must be ≥2× faster
// than the serial path.
func BenchmarkDgemmJ10Dispatch(b *testing.B) {
	var impl Implementation
	for _, sz := range []int{64, 128, 256, 512, 1024} {
		b.Run(fmt.Sprintf("n=%d", sz), func(b *testing.B) {
			rnd := rand.New(rand.NewPCG(uint64(sz), 41))
			a := make([]float64, sz*sz)
			bb := make([]float64, sz*sz)
			c := make([]float64, sz*sz)
			for i := range a {
				a[i] = rnd.NormFloat64()
				bb[i] = rnd.NormFloat64()
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				impl.Dgemm(blas.NoTrans, blas.NoTrans, sz, sz, sz, 1.0, a, sz, bb, sz, 0.0, c, sz)
			}
		})
	}
}

func benchDgemmJ10Rect(b *testing.B, m, n, k int, parallel bool) {
	rnd := rand.New(rand.NewPCG(uint64(m*1000+n*100+k), 23))
	a := make([]float64, m*k)
	bb := make([]float64, k*n)
	c := make([]float64, m*n)
	for i := range a {
		a[i] = rnd.NormFloat64()
	}
	for i := range bb {
		bb[i] = rnd.NormFloat64()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if parallel {
			dgemmParallel(false, false, m, n, k, a, k, bb, n, c, n, 1)
		} else {
			dgemmSerial(false, false, m, n, k, a, k, bb, n, c, n, 1)
		}
	}
}
