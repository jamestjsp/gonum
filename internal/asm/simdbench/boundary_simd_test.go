// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package simdbench

import (
	"fmt"
	"strings"
	"testing"
)

// These sizes straddle vector widths and common unroll boundaries. The final
// two retain a tail after a long vector loop instead of measuring only exact
// multiples, as BenchmarkCurrentVsSIMD does at its large size.
var boundarySizes = []int{7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129, 4095, 4097}

func BenchmarkSIMDBoundaries(b *testing.B) {
	for _, entry := range AMD64Assembly {
		if strings.Contains(entry.Symbol, "Inc") || entry.Mode == CompositeSIMD {
			continue
		}
		for _, n := range boundarySizes {
			benchmarkKernelVariants(b, entry, n, 1)
		}
	}
}

func BenchmarkSIMDStrides(b *testing.B) {
	for _, entry := range AMD64Assembly {
		if !strings.Contains(entry.Symbol, "Inc") {
			continue
		}
		for _, n := range []int{33, 4096} {
			for _, stride := range []int{1, 2, 3, 7, 16, 63} {
				benchmarkKernelVariants(b, entry, n, stride)
			}
		}
	}
}

func benchmarkKernelVariants(b *testing.B, entry Entry, n, stride int) {
	b.Helper()
	for _, useSIMD := range []bool{false, true} {
		impl := "current"
		if useSIMD {
			impl = "simd"
		}
		b.Run(fmt.Sprintf("%s/%s/n=%d/stride=%d/implementation=%s", entry.Package, entry.Symbol, n, stride, impl), func(b *testing.B) {
			runner := newKernelRunStride(entry, n, stride, useSIMD, true)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				runner.run()
			}
			benchmarkSink = runner.result()
		})
	}
}

func TestSIMDStridedEquivalence(t *testing.T) {
	for _, entry := range AMD64Assembly {
		if !strings.Contains(entry.Symbol, "Inc") {
			continue
		}
		for _, n := range []int{0, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129} {
			for _, stride := range []int{1, 2, 3, 7, 16, 63} {
				t.Run(fmt.Sprintf("%s/%s/n=%d/stride=%d", entry.Package, entry.Symbol, n, stride), func(t *testing.T) {
					current := newKernelRunStride(entry, n, stride, false, false)
					candidate := newKernelRunStride(entry, n, stride, true, false)
					current.run()
					candidate.run()
					if !sameKernelResult(current.result(), candidate.result()) {
						t.Fatalf("current=%v SIMD=%v", current.result(), candidate.result())
					}
				})
			}
		}
	}
}
