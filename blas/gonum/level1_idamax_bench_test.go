// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"
)

var idamaxBenchmarkSink int

func BenchmarkIdamaxUnitaryPatterns(b *testing.B) {
	for _, n := range []int{15, 16, 17, 256, 4096} {
		for _, pattern := range []string{"rare", "monotone"} {
			b.Run(fmt.Sprintf("n=%d/pattern=%s", n, pattern), func(b *testing.B) {
				x := make([]float64, n)
				fillIdamaxPattern(x, pattern)
				want := 0
				if pattern == "monotone" {
					want = n - 1
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idamaxBenchmarkSink = impl.Idamax(n, x, 1)
				}
				b.StopTimer()
				if idamaxBenchmarkSink != want {
					b.Fatalf("unexpected index: got %d want %d", idamaxBenchmarkSink, want)
				}
			})
		}
	}
}

func BenchmarkIsamaxUnitaryPatterns(b *testing.B) {
	for _, n := range []int{15, 16, 17, 256, 4096} {
		for _, pattern := range []string{"rare", "monotone"} {
			b.Run(fmt.Sprintf("n=%d/pattern=%s", n, pattern), func(b *testing.B) {
				x := make([]float32, n)
				for i := range x {
					x[i] = float32(n - i)
					if pattern == "monotone" {
						x[i] = float32(i + 1)
					}
				}
				want := 0
				if pattern == "monotone" {
					want = n - 1
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					idamaxBenchmarkSink = impl.Isamax(n, x, 1)
				}
				b.StopTimer()
				if idamaxBenchmarkSink != want {
					b.Fatalf("unexpected index: got %d want %d", idamaxBenchmarkSink, want)
				}
			})
		}
	}
}

func fillIdamaxPattern(x []float64, pattern string) {
	for i := range x {
		x[i] = float64(len(x) - i)
		if pattern == "monotone" {
			x[i] = float64(i + 1)
		}
	}
}
