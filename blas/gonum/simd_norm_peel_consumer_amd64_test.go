// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"testing"
)

func normPeelConsumerFixtures() []simdConsumerFixture {
	var fixtures []simdConsumerFixture
	impl := Implementation{}
	for _, n := range []int{15, 16, 17, 31, 32, 33, 64, 4096} {
		for _, inc := range []int{1, 2, 3} {
			x := make([]float64, (n-1)*inc+1)
			for i := range x {
				x[i] = math.NaN()
			}
			var numerator int64
			for i := 0; i < n; i++ {
				k := int64(i%17 - 8)
				x[i*inc] = float64(k) / 16
				numerator += k * k
			}
			// Integer component squares and their total are exact. The maximum
			// absolute input is 1/2, so the scaled ASM recurrence is exact too.
			want := math.Sqrt(float64(numerator) / 256)
			original := make([]uint64, len(x))
			for i, v := range x {
				original[i] = math.Float64bits(v)
			}
			var got float64
			fixtures = append(fixtures, simdConsumerFixture{
				name: fmt.Sprintf("Dnrm2/n=%d/inc=%d", n, inc),
				run: func() {
					simdConsumerClearAVX()
					got = impl.Dnrm2(n, x, inc)
				},
				check: func(t testing.TB) {
					if math.Float64bits(got) != math.Float64bits(want) {
						t.Fatalf("norm n=%d inc=%d got=%x want=%x", n, inc, math.Float64bits(got), math.Float64bits(want))
					}
					for i, v := range x {
						if math.Float64bits(v) != original[i] {
							t.Fatalf("norm input/gap changed at %d", i)
						}
					}
				},
			})
		}
	}
	return fixtures
}

func TestNormPeelConsumerFixtures(t *testing.T) {
	for _, f := range normPeelConsumerFixtures() {
		t.Run(f.name, func(t *testing.T) {
			for range 3 {
				f.run()
				f.check(t)
			}
		})
	}
}

func BenchmarkNormPeelConsumers(b *testing.B) {
	for _, f := range normPeelConsumerFixtures() {
		b.Run(f.name, func(b *testing.B) {
			f.run()
			f.check(b)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f.run()
			}
			b.StopTimer()
			f.check(b)
		})
	}
}
