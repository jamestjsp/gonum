// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"simd/archsimd"
	"slices"
	"testing"
)

type simdWidenedDotConsumerFixture struct {
	name  string
	run   func()
	check func(testing.TB)
}

func TestSIMDWidenedDotConsumerFixtures(t *testing.T) {
	for _, fixture := range simdWidenedDotConsumerFixtures() {
		t.Run(fixture.name, func(t *testing.T) {
			for range 3 {
				fixture.run()
				fixture.check(t)
			}
		})
	}
}

func BenchmarkSIMDWidenedDotConsumers(b *testing.B) {
	for _, fixture := range simdWidenedDotConsumerFixtures() {
		b.Run(fixture.name, func(b *testing.B) {
			fixture.run()
			fixture.check(b)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fixture.run()
			}
			b.StopTimer()
			fixture.check(b)
		})
	}
}

func simdWidenedDotConsumerClearAVX() {
	if archsimd.X86.AVX() {
		archsimd.ClearAVXUpperBits()
	}
}

func simdWidenedDotConsumerFixtures() []simdWidenedDotConsumerFixture {
	var fixtures []simdWidenedDotConsumerFixture
	impl := Implementation{}
	for _, n := range []int{7, 15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65, 127, 129, 4096, 4097} {
		for _, inc := range []int{1, 3} {
			x, y := make([]float32, (n-1)*inc+1), make([]float32, (n-1)*inc+1)
			for i := range x {
				x[i], y[i] = float32(i%11-5)/16, float32(i%13-6)/8
			}
			originalX, originalY := slices.Clone(x), slices.Clone(y)
			// Bounded dyadic products and sums are exactly representable, so
			// this independent scalar oracle does not hide rounding errors.
			var want float64
			for i := 0; i < n; i++ {
				want += float64(x[i*inc]) * float64(y[i*inc])
			}
			var result64 float64
			var result32 float32
			const alpha float32 = 0.25
			fixtures = append(fixtures, simdWidenedDotConsumerFixture{
				name: fmt.Sprintf("Dsdot/n=%d/inc=%d", n, inc),
				run: func() {
					simdWidenedDotConsumerClearAVX()
					result64 = impl.Dsdot(n, x, inc, y, inc)
				},
				check: func(t testing.TB) {
					if result64 != want {
						t.Fatalf("got=%v want=%v", result64, want)
					}
					if !slices.Equal(x, originalX) || !slices.Equal(y, originalY) {
						t.Fatal("Dsdot modified an input")
					}
				},
			}, simdWidenedDotConsumerFixture{
				name: fmt.Sprintf("Sdsdot/n=%d/inc=%d", n, inc),
				run: func() {
					simdWidenedDotConsumerClearAVX()
					result32 = impl.Sdsdot(n, alpha, x, inc, y, inc)
				},
				check: func(t testing.TB) {
					want32 := float32(float64(alpha) + want)
					if result32 != want32 {
						t.Fatalf("got=%v want=%v", result32, want32)
					}
					if !slices.Equal(x, originalX) || !slices.Equal(y, originalY) {
						t.Fatal("Sdsdot modified an input")
					}
				},
			})
		}
	}
	return fixtures
}
