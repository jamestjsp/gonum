// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"testing"
)

func TestIndexMaxStridedOrder(t *testing.T) {
	for _, n := range []int{1, 4, 15, 31, 32, 33, 64, 65, 257} {
		for _, inc := range []int{1, 2, 3, 8, 257} {
			for _, offset := range []int{0, 1} {
				for pattern := 0; pattern < 7; pattern++ {
					for _, specialPos := range []int{0, 1, n / 2, n - 1} {
						x := make([]float64, offset+(n-1)*inc+1)
						xs := make([]float32, len(x))
						for i := range x {
							x[i], xs[i] = math.Inf(1), float32(math.Inf(1))
						}
						for i := 0; i < n; i++ {
							v := float64(n - i)
							switch pattern {
							case 1:
								v = float64(i)
							case 2:
								v = float64((i*71)%113 - 56)
							case 3:
								v = -7
							case 4, 5, 6:
								v = float64(i % 5)
								if i == specialPos {
									v = math.NaN()
									if pattern == 5 {
										v = math.Inf(-1)
									}
									if pattern == 6 {
										v = 9
									}
								}
							}
							x[offset+i*inc], xs[offset+i*inc] = v, float32(v)
						}
						want, largest := 0, math.Abs(x[offset])
						for i := 1; i < n; i++ {
							v := math.Abs(x[offset+i*inc])
							if v > largest {
								want, largest = i, v
							}
						}
						gotD, gotS := impl.Idamax(n, x[offset:], inc), impl.Isamax(n, xs[offset:], inc)
						if gotD != want || gotS != want {
							t.Fatalf("n=%d inc=%d offset=%d pattern=%d pos=%d: Idamax=%d Isamax=%d want=%d", n, inc, offset, pattern, specialPos, gotD, gotS, want)
						}
					}
				}
			}
		}
	}
}
