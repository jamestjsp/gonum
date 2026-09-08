// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"testing"
)

func simdDgerContiguousFixtures() []simdConsumerFixture {
	var fixtures []simdConsumerFixture
	impl := Implementation{}
	for _, shape := range [][3]int{
		{8, 16, 0},
		{8, 16, 3},
		{8, 31, 0},
		{8, 31, 3},
		{8, 32, 0},
		{8, 32, 3},
		{8, 33, 0},
		{8, 33, 3},
		{8, 48, 0},
		{8, 48, 3},
		{8, 63, 0},
		{8, 63, 3},
		{8, 64, 0},
		{8, 64, 3},
		{8, 65, 0},
		{8, 65, 3},
		{64, 16, 0},
		{64, 16, 3},
		{64, 31, 0},
		{64, 31, 3},
		{64, 32, 0},
		{64, 32, 3},
		{64, 33, 0},
		{64, 33, 3},
		{64, 48, 0},
		{64, 48, 3},
		{64, 63, 0},
		{64, 63, 3},
		{64, 64, 0},
		{64, 64, 3},
		{64, 65, 0},
		{64, 65, 3},
	} {
		m, n, pad := shape[0], shape[1], shape[2]
		const inc = 1
		lda := n + pad
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
			name:  fmt.Sprintf("m=%d/n=%d/pad=%d", m, n, pad),
			run:   func() { copy(a, original); simdConsumerClearAVX(); impl.Dger(m, n, 0.5, x, inc, y, inc, a, lda) },
			check: func(t testing.TB) { simdConsumerCheck(t, a, want) },
		})
	}
	return fixtures
}

func TestSIMDDgerContiguousFixtures(t *testing.T) {
	for _, f := range simdDgerContiguousFixtures() {
		t.Run(f.name, func(t *testing.T) {
			for range 3 {
				f.run()
				f.check(t)
			}
		})
	}
}

func BenchmarkSIMDDgerContiguous(b *testing.B) {
	for _, f := range simdDgerContiguousFixtures() {
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
