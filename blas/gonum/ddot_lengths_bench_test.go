// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"
)

var ddotLengthsSink float64

func BenchmarkDdotUnitaryLengths(b *testing.B) {
	for _, n := range []int{7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256, 480, 1024, 4096} {
		for _, offset := range []int{0, 1} {
			b.Run(fmt.Sprintf("n=%d/offset=%d", n, offset), func(b *testing.B) {
				x := make([]float64, offset+n)
				y := make([]float64, offset+n)
				for i := range x {
					x[i] = 1 + float64(i%17)/32
					y[i] = 1 + float64(i%13)/16
				}
				x, y = x[offset:], y[offset:]
				b.ReportAllocs()
				for b.Loop() {
					ddotLengthsSink = impl.Ddot(n, x, 1, y, 1)
				}
			})
		}
	}
}
