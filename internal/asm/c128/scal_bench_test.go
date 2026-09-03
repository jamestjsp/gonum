// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package c128

import (
	"fmt"
	"testing"
)

func BenchmarkScaleUnitary(b *testing.B) {
	for _, n := range []int{10, 100, 1000, 50000} {
		b.Run(fmt.Sprintf("complex-%d", n), func(b *testing.B) {
			x := x[:n]
			for i := 0; i < b.N; i++ {
				ScalUnitary(1+1i, x)
			}
		})
		b.Run(fmt.Sprintf("real-%d", n), func(b *testing.B) {
			x := x[:n]
			for i := 0; i < b.N; i++ {
				DscalUnitary(2, x)
			}
		})
	}
}
