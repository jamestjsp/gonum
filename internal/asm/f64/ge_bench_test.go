// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import (
	"fmt"
	"testing"
)

func BenchmarkGemvTStrided(b *testing.B) {
	for _, m := range []int{10, 1000} {
		for _, n := range []int{8, 10, 12, 16, 20, 24, 31, 32, 33, 1000} {
			b.Run(fmt.Sprintf("m=%d/n=%d", m, n), func(b *testing.B) {
				const incX, incY = 2, 3
				a := make([]float64, m*n)
				x := make([]float64, (m-1)*incX+1)
				y := make([]float64, (n-1)*incY+1)
				for _, v := range [][]float64{a, x, y} {
					for i := range v {
						v[i] = float64(i%31-15) / 7
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					GemvT(uintptr(m), uintptr(n), 2, a, uintptr(n), x, incX, 1, y, incY)
				}
			})
		}
	}
}
