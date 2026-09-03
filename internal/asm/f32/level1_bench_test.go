// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32

import (
	"fmt"
	"testing"
)

func BenchmarkSasumUnitary(t *testing.B) {
	benchmarkLevel1Lengths(t, func(b *testing.B, x, _ []float32) {
		for i := 0; i < b.N; i++ {
			benchSink = SasumUnitary(x)
		}
	})
}

func BenchmarkSwapUnitary(t *testing.B) {
	benchmarkLevel1Lengths(t, func(b *testing.B, x, y []float32) {
		for i := 0; i < b.N; i++ {
			SwapUnitary(x, y)
		}
	})
}

func BenchmarkRotUnitary(t *testing.B) {
	benchmarkLevel1Lengths(t, func(b *testing.B, x, y []float32) {
		for i := 0; i < b.N; i++ {
			RotUnitary(x, y, 0.5, -0.25)
		}
	})
}

func BenchmarkRotmUnitaryRescaling(t *testing.B) {
	benchmarkLevel1Lengths(t, func(b *testing.B, x, y []float32) {
		for i := 0; i < b.N; i++ {
			RotmUnitaryRescaling(x, y, 0.5, 0.25, -0.25, 0.5)
		}
	})
}

func BenchmarkRotmUnitaryOffDiagonal(t *testing.B) {
	benchmarkLevel1Lengths(t, func(b *testing.B, x, y []float32) {
		for i := 0; i < b.N; i++ {
			RotmUnitaryOffDiagonal(x, y, 0.25, -0.25)
		}
	})
}

func BenchmarkRotmUnitaryDiagonal(t *testing.B) {
	benchmarkLevel1Lengths(t, func(b *testing.B, x, y []float32) {
		for i := 0; i < b.N; i++ {
			RotmUnitaryDiagonal(x, y, 0.5, 0.5)
		}
	})
}

func benchmarkLevel1Lengths(t *testing.B, fn func(*testing.B, []float32, []float32)) {
	for _, n := range []int{10, 100, 1000, 10000, 50000} {
		t.Run(fmt.Sprint(n), func(b *testing.B) {
			x := make([]float32, n)
			y := make([]float32, n)
			b.ResetTimer()
			fn(b, x, y)
		})
	}
}
