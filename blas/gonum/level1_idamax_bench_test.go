// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"
)

var idamaxBenchmarkSink int

// BenchmarkIndexMaxStrided includes both frequent and rare index updates.
func BenchmarkIndexMaxStrided(b *testing.B) {
	for _, n := range []int{15, 16, 17, 31, 32, 33, 256, 4096} {
		for _, inc := range []int{2, 3, 17, 257} {
			for _, pattern := range []string{"rare", "monotone", "random"} {
				for _, precision := range []string{"float64", "float32"} {
					b.Run(fmt.Sprintf("n=%d/inc=%d/pattern=%s/type=%s", n, inc, pattern, precision), func(b *testing.B) {
						x := make([]float64, (n-1)*inc+1)
						xs := make([]float32, len(x))
						want, largest := 0, -1.0
						for i := 0; i < n; i++ {
							v := float64(n - i)
							if pattern == "monotone" {
								v = float64(i + 1)
							}
							if pattern == "random" {
								v = float64((i * 7919) % 1009)
							}
							x[i*inc], xs[i*inc] = v, float32(v)
							if v > largest {
								want, largest = i, v
							}
						}
						b.ReportAllocs()
						b.ResetTimer()
						if precision == "float64" {
							for i := 0; i < b.N; i++ {
								idamaxBenchmarkSink = impl.Idamax(n, x, inc)
							}
						} else {
							for i := 0; i < b.N; i++ {
								idamaxBenchmarkSink = impl.Isamax(n, xs, inc)
							}
						}
						b.StopTimer()
						if idamaxBenchmarkSink != want {
							b.Fatalf("got %d, want %d", idamaxBenchmarkSink, want)
						}
					})
				}
			}
		}
	}
}

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
