// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"
)

func TestSIMDStridedGerAliasing(t *testing.T) {
	for _, alias := range []string{"x", "y", "rows"} {
		const m, n, inc = 8, 9, 3
		lda := n + 3
		if alias == "rows" {
			lda = 4
		}
		a, _ := matrixVector((m-1)*lda+n, 1)
		want := slices.Clone(a)
		x, _ := matrixVector(m, inc)
		y, _ := matrixVector(n, inc)
		wx, wy := x, y
		if alias == "x" {
			x, wx = a[1:1+(m-1)*inc+1], want[1:1+(m-1)*inc+1]
		}
		if alias == "y" {
			y, wy = a[2:2+(n-1)*inc+1], want[2:2+(n-1)*inc+1]
		}
		for i := 0; i < m; i++ {
			scale := float32(-0.75) * wx[i*inc]
			for j := 0; j < n; j++ {
				want[i*lda+j] += scale * wy[j*inc]
			}
		}
		GerSIMD(m, n, -0.75, x, inc, y, inc, a, uintptr(lda))
		for i, value := range a {
			if value != want[i] {
				t.Fatalf("alias%s index%d: got%v want%v", alias, i, value, want[i])
			}
		}
	}
}

func TestSIMDStridedGerExceptional(t *testing.T) {
	values := []float32{0, math.Float32frombits(1 << 31), 1, -1, math.MaxFloat32, -math.MaxFloat32, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN())}
	for _, n := range []int{4, 7, 8, 9, 17} {
		const m, inc = 9, 3
		for _, alpha := range []float32{0, -0.75, math.MaxFloat32, float32(math.Inf(1))} {
			x, y := make([]float32, (m-1)*inc+1), make([]float32, (n-1)*inc+1)
			for i := 0; i < m; i++ {
				x[i*inc] = values[i%len(values)]
			}
			for i := 0; i < n; i++ {
				y[i*inc] = values[i%len(values)]
			}
			a, want := make([]float32, m*n), make([]float32, m*n)
			for i := 0; i < m; i++ {
				scale := alpha * x[i*inc]
				for j := 0; j < n; j++ {
					want[i*n+j] += scale * y[j*inc]
				}
			}
			GerSIMD(m, uintptr(n), alpha, x, inc, y, inc, a, uintptr(n))
			for i, value := range a {
				if math.Float32bits(value) != math.Float32bits(want[i]) && !(math.IsNaN(float64(value)) && math.IsNaN(float64(want[i]))) {
					t.Fatalf("n%d alpha%v index%d: got%v want%v", n, alpha, i, value, want[i])
				}
			}
		}
	}
}

func TestSIMDStridedGerIndependentGaps(t *testing.T) {
	for _, n := range []int{8, 31, 64, 65} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			const m, inc = 8, 3
			x, y := make([]float32, (m-1)*inc+1), make([]float32, (n-1)*inc+1)
			for i := 0; i < m; i++ {
				x[i*inc] = 1
			}
			for i := 0; i < n; i++ {
				y[i*inc] = 2
			}
			done := make(chan struct{})
			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					select {
					case <-done:
						return
					default:
						for i := range x {
							if i%inc != 0 {
								x[i]++
							}
						}
						for i := range y {
							if i%inc != 0 {
								y[i]++
							}
						}
					}
				}
			}()
			a := make([]float32, m*n)
			for i := 0; i < 100; i++ {
				GerSIMD(m, uintptr(n), 0.5, x, inc, y, inc, a, uintptr(n))
			}
			close(done)
			wg.Wait()
			for i, value := range a {
				if value != 100 {
					t.Fatalf("index%d: got%v want100", i, value)
				}
			}
		})
	}
}
