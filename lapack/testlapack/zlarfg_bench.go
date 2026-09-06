// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// ZlarfgBenchmark includes input restoration so every call uses the same data.
func ZlarfgBenchmark(b *testing.B, impl Zlarfger) {
	for _, n := range []int{16, 32, 33, 64, 256, 4096} {
		for _, inc := range []int{1, 2} {
			b.Run(fmt.Sprintf("n=%d/inc=%d", n, inc), func(b *testing.B) {
				rnd := rand.New(rand.NewPCG(1, 2))
				input := make([]complex128, (n-2)*inc+1)
				for i := range input {
					input[i] = complex(rnd.NormFloat64(), rnd.NormFloat64())
				}
				x := make([]complex128, len(input))
				var beta float64
				var tau complex128
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(x, input)
					beta, tau = impl.Zlarfg(n, 1+2i, x, inc)
				}
				b.StopTimer()
				if math.IsNaN(beta) || math.IsInf(beta, 0) || math.IsNaN(real(tau)) || math.IsNaN(imag(tau)) {
					b.Fatal("non-finite reflector")
				}
			})
		}
	}
}
