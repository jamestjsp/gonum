// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64

import (
	"fmt"
	"testing"
)

var level1BenchSink float64

func BenchmarkDasumUnitary(b *testing.B) {
	benchmarkLevel1Lengths(b, func(b *testing.B, x, _ []float64) {
		for i := 0; i < b.N; i++ {
			level1BenchSink = DasumUnitary(x)
		}
	})
}

func BenchmarkSwapUnitary(b *testing.B) {
	benchmarkLevel1Lengths(b, func(b *testing.B, x, y []float64) {
		for i := 0; i < b.N; i++ {
			SwapUnitary(x, y)
		}
	})
}

func BenchmarkRotUnitary(b *testing.B) {
	benchmarkLevel1Lengths(b, func(b *testing.B, x, y []float64) {
		for i := 0; i < b.N; i++ {
			RotUnitary(x, y, 0.5, -0.25)
		}
	})
}

func BenchmarkRotmUnitaryRescaling(b *testing.B) {
	benchmarkLevel1Lengths(b, func(b *testing.B, x, y []float64) {
		for i := 0; i < b.N; i++ {
			RotmUnitaryRescaling(x, y, 0.5, 0.25, -0.25, 0.5)
		}
	})
}

func BenchmarkRotmUnitaryOffDiagonal(b *testing.B) {
	benchmarkLevel1Lengths(b, func(b *testing.B, x, y []float64) {
		for i := 0; i < b.N; i++ {
			RotmUnitaryOffDiagonal(x, y, 0.25, -0.25)
		}
	})
}

func BenchmarkRotmUnitaryDiagonal(b *testing.B) {
	benchmarkLevel1Lengths(b, func(b *testing.B, x, y []float64) {
		for i := 0; i < b.N; i++ {
			RotmUnitaryDiagonal(x, y, 0.5, 0.5)
		}
	})
}

func benchmarkLevel1Lengths(b *testing.B, fn func(*testing.B, []float64, []float64)) {
	for _, n := range []int{10, 100, 1000, 10000, 50000} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			x := make([]float64, n)
			y := make([]float64, n)
			b.ResetTimer()
			fn(b, x, y)
		})
	}
}
