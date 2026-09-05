// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"testing"
)

// Include the short-tail crossover as well as exact vector multiples, which
// expose call-frame overhead even when no tail helper executes.
func BenchmarkSIMDAxpyTails(b *testing.B) {
	for _, to := range []bool{false, true} {
		for _, n := range []int{7, 8, 11, 12, 15, 16, 31, 32, 64} {
			b.Run(fmt.Sprintf("to=%t/n=%d", to, n), func(b *testing.B) {
				x, y, dst := make([]float32, n), make([]float32, n), make([]float32, n)
				for i := range x {
					x[i] = float32(i%11-5) * 0.125
					y[i] = float32(i%7-3) * 0.25
				}
				var run func()
				if to {
					run = func() { AxpyUnitaryToSIMD(dst, 0.75, x, y) }
				} else {
					run = func() { AxpyUnitarySIMD(0.75, x, y) }
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run()
				}
			})
		}
	}
}
