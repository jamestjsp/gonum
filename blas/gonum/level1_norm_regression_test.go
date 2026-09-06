// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"math/big"
	"testing"
)

func TestSingleNormWidenedRange(t *testing.T) {
	for _, n := range []int{0, 1, 3, 4, 7, 15, 16, 17, 31, 32, 33, 255, 256, 257, 4097} {
		for _, inc := range []int{1, 2, 3, 17} {
			for _, scale := range []float32{math.SmallestNonzeroFloat32, 1e-40, 1e-20, 1, 1e20, math.MaxFloat32 / 128, math.MaxFloat32} {
				t.Run(fmt.Sprintf("n=%d/inc=%d/scale=%g", n, inc, scale), func(t *testing.T) {
					x := make([]float32, max(0, (n-1)*inc+1))
					z := make([]complex64, len(x))
					for i := range x {
						x[i], z[i] = float32(math.NaN()), complex(float32(math.Inf(1)), float32(math.NaN()))
					}
					realSum, complexSum := new(big.Float).SetPrec(256), new(big.Float).SetPrec(256)
					for i := 0; i < n; i++ {
						re := scale * float32(0.25+0.5*math.Sin(float64(i)))
						im := scale * float32(0.25+0.5*math.Cos(float64(i)))
						if i%7 == 0 {
							re = math.SmallestNonzeroFloat32
						}
						x[i*inc], z[i*inc] = re, complex(re, im)
						for j, v := range []float32{re, im} {
							f := new(big.Float).SetPrec(256).SetFloat64(float64(v))
							sq := new(big.Float).SetPrec(256).Mul(f, f)
							complexSum.Add(complexSum, sq)
							if j == 0 {
								realSum.Add(realSum, sq)
							}
						}
					}
					wantReal, _ := realSum.Sqrt(realSum).Float32()
					wantComplex, _ := complexSum.Sqrt(complexSum).Float32()
					checkSingleNorm(t, impl.Snrm2(n, x, inc), wantReal, 2e-7)
					tol := 2e-7
					if n < 32 {
						tol = 3e-6 // The small complex path retains its float32 scaled recurrence.
					}
					checkSingleNorm(t, impl.Scnrm2(n, z, inc), wantComplex, tol)
				})
			}
		}
	}
}

func checkSingleNorm(t *testing.T, got, want float32, tol float64) {
	t.Helper()
	if math.IsInf(float64(want), 1) {
		if !math.IsInf(float64(got), 1) {
			t.Fatalf("got %g, want +Inf", got)
		}
		return
	}
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || math.Abs(float64(got)-float64(want)) > tol*float64(want)+2*math.SmallestNonzeroFloat32 {
		t.Fatalf("got %g, want %g", got, want)
	}
}

func TestSingleNormSpecialValues(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	for _, n := range []int{1, 4, 31, 32, 33, 256, 257} {
		for _, inc := range []int{1, 2, 3} {
			for _, pair := range [][2]float32{{nan, 1}, {inf, 1}, {nan, inf}, {inf, nan}, {inf, inf}} {
				x := make([]float32, (n-1)*inc+1)
				z := make([]complex64, len(x))
				for i := range x {
					x[i], z[i] = nan, complex(inf, nan)
				}
				for i := 0; i < n; i++ {
					x[i*inc], z[i*inc] = 1, 1
				}
				x[0] = pair[0]
				if n > 1 {
					x[(n-1)*inc] = pair[1]
				}
				z[(n-1)*inc] = complex(pair[0], pair[1])
				gotReal := impl.Snrm2(n, x, inc)
				wantNaN := math.IsNaN(float64(pair[0])) || n > 1 && math.IsNaN(float64(pair[1]))
				if wantNaN && !math.IsNaN(float64(gotReal)) || !wantNaN && !math.IsInf(float64(gotReal), 1) {
					t.Fatalf("real n=%d inc=%d pair=%v: got %g", n, inc, pair, gotReal)
				}
				gotComplex := impl.Scnrm2(n, z, inc)
				wantInf := math.IsInf(float64(pair[0]), 0) || math.IsInf(float64(pair[1]), 0)
				if wantInf && !math.IsInf(float64(gotComplex), 1) || !wantInf && !math.IsNaN(float64(gotComplex)) {
					t.Fatalf("complex n=%d inc=%d pair=%v: got %g", n, inc, pair, gotComplex)
				}
			}
			x := make([]float32, (n-1)*inc+1)
			z := make([]complex64, len(x))
			for i := range x {
				x[i], z[i] = float32(math.Copysign(0, -1)), complex(float32(math.Copysign(0, -1)), 0)
			}
			for _, got := range []float32{impl.Snrm2(n, x, inc), impl.Scnrm2(n, z, inc)} {
				if got != 0 || math.Signbit(float64(got)) {
					t.Fatalf("n=%d inc=%d: got %g, want +0", n, inc, got)
				}
			}
		}
	}
}
