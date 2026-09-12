// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"
)

// BenchmarkDggesNetlib compares preallocated Go and reference LAPACK drivers.
// Each uses its native storage layout; input copies are timed for both, while
// layout conversion and workspace queries are excluded. Go allocation counters
// cannot account for C allocations. The Netlib *_work interface reuses scratch.
func BenchmarkDggesNetlib(b *testing.B) {
	for _, n := range []int{10, 50, 100, 200} {
		for _, kind := range dggesPencilKinds {
			a, bm := dggesComparisonPencil(kind, n)
			for _, vectors := range []string{"none", "right", "both"} {
				for _, sorting := range []bool{false, true} {
					for _, native := range []bool{false, true} {
						backend := "Go"
						if native {
							backend = "Netlib"
						}
						b.Run(fmt.Sprintf("n=%d/%s/vectors=%s/sort=%t/%s", n, kind, vectors, sorting, backend), func(b *testing.B) {
							c := newDggesComparison(n, a, bm, vectors, dggesSelection(kind, sorting), native)
							sdim, ok := c.run()
							if !ok {
								b.Fatal("Dgges failed")
							}
							if sorting && (sdim == 0 || sdim == n) {
								b.Fatal("fixture does not split spectrum")
							}
							b.ReportAllocs()
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								got, ok := c.run()
								if !ok || got != sdim {
									b.Fatalf("sdim=%d, ok=%t; want %d, true", got, ok, sdim)
								}
							}
						})
					}
				}
			}
		}
	}
}
