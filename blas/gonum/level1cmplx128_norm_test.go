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

func TestDznrm2MagnitudeBoundaries(t *testing.T) {
	for _, n := range []int{0, 1, 4, 31, 32, 33, 65, 257} {
		for _, inc := range []int{1, 2} {
			for _, scale := range []float64{math.SmallestNonzeroFloat64, 1e-310, 1e-150, 1, 1e150, 1e300} {
				t.Run(fmt.Sprintf("n=%d/inc=%d/scale=%g", n, inc, scale), func(t *testing.T) {
					x := make([]complex128, max(0, (n-1)*inc+1))
					for i := range x {
						x[i] = complex(math.NaN(), math.NaN())
					}
					sum := new(big.Float).SetPrec(256)
					for i := 0; i < n; i++ {
						x[i*inc] = complex(scale*(0.25+math.Sin(float64(i))), scale*(0.5+math.Cos(float64(i))))
						for _, v := range []float64{real(x[i*inc]), imag(x[i*inc])} {
							f := new(big.Float).SetPrec(256).SetFloat64(v)
							sum.Add(sum, new(big.Float).SetPrec(256).Mul(f, f))
						}
					}
					want, _ := sum.Sqrt(sum).Float64()
					got := (Implementation{}).Dznrm2(n, x, inc)
					if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 3e-14*want+2*math.SmallestNonzeroFloat64 {
						t.Fatalf("got %g, want %g", got, want)
					}
				})
			}
		}
	}
}

func TestComplexNormSpecialValuePriority(t *testing.T) {
	for _, n := range []int{1, 31, 32, 33, 65} {
		for _, inc := range []int{1, 2} {
			for _, special := range []complex128{complex(math.NaN(), 1), complex(math.Inf(1), math.NaN()), complex(math.NaN(), math.Inf(-1)), complex(math.Inf(1), 0)} {
				for _, pos := range []int{0, n - 1} {
					x := make([]complex128, (n-1)*inc+1)
					xs := make([]complex64, len(x))
					for i := 0; i < n; i++ {
						x[i*inc], xs[i*inc] = 1+2i, 1+2i
					}
					x[pos*inc], xs[pos*inc] = special, complex64(special)
					wantInf := math.IsInf(real(special), 0) || math.IsInf(imag(special), 0)
					for _, got := range []float64{(Implementation{}).Dznrm2(n, x, inc), float64((Implementation{}).Scnrm2(n, xs, inc))} {
						if wantInf && !math.IsInf(got, 1) || !wantInf && !math.IsNaN(got) {
							t.Fatalf("n=%d inc=%d pos=%d special=%v: got %g", n, inc, pos, special, got)
						}
					}
				}
			}
		}
	}
	for _, got := range []float64{(Implementation{}).Dznrm2(65, make([]complex128, 65), 1), float64((Implementation{}).Scnrm2(65, make([]complex64, 65), 1))} {
		if got != 0 || math.Signbit(got) {
			t.Fatalf("zero norm: got %g, want +0", got)
		}
	}
}
