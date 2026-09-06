// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import "testing"

func Dtgex2ScratchBenchmark(b *testing.B, impl Dtgex2er) {
	aOrig := []float64{
		2, 1, 0.1, 0.2,
		-1, 2, 0.2, 0.1,
		0, 0, 3, 1,
		0, 0, -1, 3,
	}
	bOrig := []float64{
		1, 0, 0.05, 0.02,
		0, 1, 0.1, 0.05,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	a := make([]float64, 16)
	bm := make([]float64, 16)
	q := make([]float64, 16)
	z := make([]float64, 16)
	work := make([]float64, 40)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		copy(a, aOrig)
		copy(bm, bOrig)
		clear(q)
		clear(z)
		for i := 0; i < 4; i++ {
			q[i*4+i] = 1
			z[i*4+i] = 1
		}
		if !impl.Dtgex2(true, true, 4, a, 4, bm, 4, q, 4, z, 4, 0, 2, 2, work, len(work)) {
			b.Fatal("Dtgex2 swap failed")
		}
	}
}
