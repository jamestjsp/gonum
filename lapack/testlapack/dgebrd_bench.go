// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// DgebrdBenchmark measures bidiagonal reduction with reused workspace. Restoring
// the input matrix is included in the measured operation.
func DgebrdBenchmark(b *testing.B, impl Dgebrder) {
	for _, shape := range [][2]int{{16, 16}, {64, 64}, {128, 128}, {256, 256}, {512, 64}, {64, 512}, {256, 128}, {128, 256}} {
		m, n := shape[0], shape[1]
		b.Run(fmt.Sprintf("m=%d/n=%d", m, n), func(b *testing.B) {
			rnd := rand.New(rand.NewPCG(1, 1))
			lda, minmn := n+3, min(m, n)
			original, a := make([]float64, m*lda), make([]float64, m*lda)
			for i := range original {
				original[i] = rnd.NormFloat64()
			}
			d, e := make([]float64, minmn), make([]float64, minmn-1)
			tauQ, tauP := make([]float64, minmn), make([]float64, minmn)
			work := make([]float64, 1)
			impl.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, -1)
			work = make([]float64, int(work[0]))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(a, original)
				impl.Dgebrd(m, n, a, lda, d, e, tauQ, tauP, work, len(work))
			}
		})
	}
}
