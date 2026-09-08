// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"testing"
)

func simdDgerShapeFixtures() []simdConsumerFixture {
	var fixtures []simdConsumerFixture
	impl := Implementation{}
	for _, shape := range [][3]int{
		{3, 8, 1}, {4, 8, 1}, {5, 8, 1}, {8, 8, 1}, {64, 8, 1}, {65, 8, 1}, {8, 7, 1}, {8, 9, 1},
		{4, 31, 3}, {4, 64, 7}, {8, 16, 2}, {8, 31, 7},
		{8, 63, 3}, {16, 32, 1}, {16, 48, 3}, {16, 64, 7},
		{64, 16, 3}, {64, 31, 2}, {64, 48, 7}, {64, 63, 3},
		{64, 64, 2}, {64, 65, 7}, {128, 32, 1}, {128, 64, 3},
		{8, 129, 3}, {64, 129, 7}, {65, 31, 2}, {128, 129, 3},
	} {
		m, n, inc := shape[0], shape[1], shape[2]
		lda := n + 3
		x, y := make([]float64, (m-1)*inc+1), make([]float64, (n-1)*inc+1)
		a := make([]float64, (m-1)*lda+n)
		for i := range x {
			x[i] = -99
		}
		for i := range y {
			y[i] = -99
		}
		for i := range a {
			a[i] = -99
		}
		for i := 0; i < m; i++ {
			x[i*inc] = float64((i%11)-5) * 0.125
			for j := 0; j < n; j++ {
				a[i*lda+j] = float64((i+j)%13-6) * 0.0625
			}
		}
		for j := 0; j < n; j++ {
			y[j*inc] = float64(j%7-3) * 0.25
		}
		original := append([]float64(nil), a...)
		want := make([]float64, len(a))
		for i, v := range a {
			want[i] = float64(v)
		}
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				want[i*lda+j] += 0.5 * float64(x[i*inc]) * float64(y[j*inc])
			}
		}
		fixtures = append(fixtures, simdConsumerFixture{
			name:  fmt.Sprintf("m=%d/n=%d/inc=%d", m, n, inc),
			run:   func() { copy(a, original); simdConsumerClearAVX(); impl.Dger(m, n, 0.5, x, inc, y, inc, a, lda) },
			check: func(t testing.TB) { simdConsumerCheck(t, a, want) },
		})
	}
	return fixtures
}

func TestSIMDDgerShapeFixtures(t *testing.T) {
	for _, f := range simdDgerShapeFixtures() {
		t.Run(f.name, func(t *testing.T) {
			for range 3 {
				f.run()
				f.check(t)
			}
		})
	}
}

func BenchmarkSIMDDgerShapes(b *testing.B) {
	for _, f := range simdDgerShapeFixtures() {
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
