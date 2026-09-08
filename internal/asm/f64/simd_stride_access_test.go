// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"math/bits"
	"slices"
	"sync"
	"testing"
)

func TestSIMDPositiveStridePanics(t *testing.T) {
	// Validation must not wrap an oversized span into a valid pointer loop.
	for _, inc := range []uintptr{2, ^uintptr(0), uintptr(1) << (bits.UintSize - 1)} {
		for _, name := range []string{"Scal", "ScalTo", "Axpy", "AxpyTo"} {
			x, y, dst := []float64{1, 2, 3, 4, 5}, []float64{6, 7, 8, 9, 10}, []float64{11, 12, 13, 14, 15}
			wantX, wantY, wantDst := slices.Clone(x), slices.Clone(y), slices.Clone(dst)
			scalar := func() {
				for i, index := uintptr(0), uintptr(0); i < 5; i, index = i+1, index+inc {
					switch name {
					case "Scal":
						wantX[index] *= 0.5
					case "ScalTo":
						wantDst[index] = 0.5 * wantX[index]
					case "Axpy":
						wantY[index] += 0.5 * wantX[index]
					case "AxpyTo":
						wantDst[index] = 0.5*wantX[index] + wantY[index]
					}
				}
			}
			candidate := func() {
				switch name {
				case "Scal":
					ScalIncSIMD(0.5, x, 5, inc)
				case "ScalTo":
					ScalIncToSIMD(dst, inc, 0.5, x, 5, inc)
				case "Axpy":
					AxpyIncSIMD(0.5, x, y, 5, inc, inc, 0, 0)
				case "AxpyTo":
					AxpyIncToSIMD(dst, inc, 0, 0.5, x, y, 5, inc, inc, 0, 0)
				}
			}
			panics := func(fn func()) (yes bool) { defer func() { yes = recover() != nil }(); fn(); return }
			if !panics(scalar) || !panics(candidate) {
				t.Fatalf("%s inc=%d: expected checked panic", name, inc)
			}
			if !slices.Equal(x, wantX) || !slices.Equal(y, wantY) || !slices.Equal(dst, wantDst) {
				t.Fatalf("%s inc=%d: changed writes before panic", name, inc)
			}
		}
	}
}

func TestSIMDIndependentStridedOutputs(t *testing.T) {
	const n = 129
	x, y, dst := make([]float64, 2*n), make([]float64, 2*n), make([]float64, 2*n)
	for i := range x {
		x[i], y[i] = 1, 2
	}
	var wg sync.WaitGroup
	for start := uintptr(0); start < 2; start++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 20 {
				AxpyIncToSIMD(dst, 2, start, 0.5, x, y, n, 2, 2, start, start)
			}
		}()
	}
	wg.Wait()
	for i, v := range dst {
		if v != 2.5 {
			t.Fatalf("index=%d got%g", i, v)
		}
	}
	// The read-only norm must not access independently mutable increment gaps.
	wg.Add(2)
	go func() {
		defer wg.Done()
		for range 20 {
			if got := L1NormIncSIMD(x, n, 2); got != n {
				t.Errorf("L1 got%g", got)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for k := range 20 {
			for i := 1; i < len(x); i += 2 {
				x[i] = math.Float64frombits(uint64(k))
			}
		}
	}()
	wg.Wait()
}
