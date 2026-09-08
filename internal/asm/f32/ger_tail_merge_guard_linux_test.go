// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import (
	"fmt"
	"runtime/debug"
	"testing"
)

func TestSIMDGerTailMergeGuarded(t *testing.T) {
	restore := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(restore)
	for _, m := range []int{8, 65} {
		for _, n := range []int{21, 22, 23, 29, 30, 31, 61, 62, 63, 69, 70, 71, 125, 126, 127} {
			for _, inc := range []int{2, 7} {
				t.Run(fmt.Sprintf("m=%d/n=%d/inc=%d", m, n, inc), func(t *testing.T) {
					lda := n + 3
					x, y, a := guardedSIMDF32(t, (m-1)*inc+1), guardedSIMDF32(t, (n-1)*inc+1), guardedSIMDF32(t, (m-1)*lda+n)
					for i := range x {
						x[i] = -99
					}
					for i := range y {
						y[i] = -99
					}
					for i := range a {
						a[i] = -99
					}
					for i := 0; i < m; i++ {
						x[i*inc] = 1
						for j := 0; j < n; j++ {
							a[i*lda+j] = 0
						}
					}
					for j := 0; j < n; j++ {
						y[j*inc] = 2
					}
					GerSIMD(uintptr(m), uintptr(n), .5, x, uintptr(inc), y, uintptr(inc), a, uintptr(lda))
					for i, v := range a {
						want := float32(-99)
						if i%lda < n {
							want = 1
						}
						if v != want {
							t.Fatalf("A[%d]=%g want%g", i, v, want)
						}
					}
				})
			}
		}
	}
}
