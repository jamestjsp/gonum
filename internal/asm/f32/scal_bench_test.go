// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32

import (
	"fmt"
	"testing"
)

func BenchmarkScalUnitary(t *testing.B) {
	for _, n := range []int{10, 100, 1000, 10000, 50000} {
		t.Run(fmt.Sprint(n), func(b *testing.B) {
			x := make([]float32, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ScalUnitary(1.0001, x)
			}
		})
	}
}

func BenchmarkScalUnitaryTo(t *testing.B) {
	for _, n := range []int{10, 100, 1000, 10000, 50000} {
		t.Run(fmt.Sprint(n), func(b *testing.B) {
			x := make([]float32, n)
			dst := make([]float32, n)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ScalUnitaryTo(dst, 1.0001, x)
			}
		})
	}
}
