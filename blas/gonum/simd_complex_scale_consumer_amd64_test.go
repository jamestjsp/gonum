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

type simdComplexScaleConsumerFixture struct {
	name  string
	run   func()
	reset func()
	check func(testing.TB, int)
}

func TestSIMDComplexScaleConsumerFixtures(t *testing.T) {
	for _, fixture := range simdComplexScaleConsumerFixtures() {
		t.Run(fixture.name, func(t *testing.T) {
			fixture.reset()
			for calls := 1; calls <= 9; calls++ {
				fixture.run()
				fixture.check(t, calls)
			}
		})
	}
}

func BenchmarkSIMDComplexScaleConsumers(b *testing.B) {
	for _, fixture := range simdComplexScaleConsumerFixtures() {
		b.Run(fixture.name, func(b *testing.B) {
			fixture.reset()
			fixture.run()
			fixture.check(b, 1)
			fixture.reset()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fixture.run()
			}
			b.StopTimer()
			fixture.check(b, b.N)
		})
	}
}

func simdComplexScaleConsumerClearAVX() {
	if archsimd.X86.AVX() {
		archsimd.ClearAVXUpperBits()
	}
}

func simdComplexScaleConsumerFixtures() []simdComplexScaleConsumerFixture {
	var fixtures []simdComplexScaleConsumerFixture
	impl := Implementation{}
	for _, n := range []int{1, 4, 5, 8, 9, 31, 32, 33, 4096} {
		for _, inc := range []int{1, 2, 3, 7} {
			for _, realScale := range []bool{false, true} {
				x := make([]complex128, (n-1)*inc+1)
				for i := range x {
					x[i] = complex(float64(i%11-5)/16, float64(i%7-3)/8)
				}
				original := slices.Clone(x)
				name := "Zscal"
				if realScale {
					name = "Zdscal"
				}
				fixtures = append(fixtures, simdComplexScaleConsumerFixture{
					name: fmt.Sprintf("%s/n=%d/inc=%d", name, n, inc),
					run: func() {
						// Exact roots of unity cycle finite dyadic inputs without
						// drift, overflow, subnormals or timed restoration copies.
						if realScale {
							simdComplexScaleConsumerClearAVX()
							impl.Zdscal(n, -1, x, inc)
						} else {
							simdComplexScaleConsumerClearAVX()
							impl.Zscal(n, 1i, x, inc)
						}
					},
					reset: func() { copy(x, original) },
					check: func(t testing.TB, calls int) {
						phase := uint(calls)
						for i, value := range original {
							want := value
							if i%inc == 0 {
								if realScale {
									if phase&1 != 0 {
										want = -value
									}
								} else {
									// Independent permutation/sign oracle, without
									// repeating the complex multiply under test.
									switch phase & 3 {
									case 1:
										want = complex(-imag(value), real(value))
									case 2:
										want = -value
									case 3:
										want = complex(imag(value), -real(value))
									}
								}
							}
							if x[i] != want {
								t.Fatalf("phase=%d index=%d got=%v want=%v", phase, i, x[i], want)
							}
						}
					},
				})
			}
		}
	}
	return fixtures
}
