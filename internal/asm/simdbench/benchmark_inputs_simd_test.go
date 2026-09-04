// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package simdbench

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func TestBenchmarkInputsStayNormal(t *testing.T) {
	for _, entry := range []Entry{
		{Package: "f64", Symbol: "Div"},
		{Package: "f64", Symbol: "ScalUnitary"},
		{Package: "f64", Symbol: "ScalInc"},
		{Package: "c128", Symbol: "DscalUnitary"},
		{Package: "c128", Symbol: "DscalInc"},
		{Package: "c128", Symbol: "ScalUnitary"},
		{Package: "c128", Symbol: "ScalInc"},
	} {
		for _, useSIMD := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/%s/simd=%t", entry.Package, entry.Symbol, useSIMD), func(t *testing.T) {
				initial := newKernelRun(entry, 33, useSIMD, true).result()
				runner := newKernelRun(entry, 33, useSIMD, true)
				for range 8192 {
					runner.run()
				}
				checkNormalBenchmarkResult(t, runner.result())
				if !reflect.DeepEqual(initial, runner.result()) {
					t.Fatal("repeated benchmark inputs drifted after complete scale cycles")
				}
				runner.run()
				checkNormalBenchmarkResult(t, runner.result())
				if reflect.DeepEqual(initial, runner.result()) {
					t.Fatal("benchmark operation did not change its inputs")
				}
			})
		}
	}
}

func checkNormalBenchmarkResult(t *testing.T, result any) {
	t.Helper()
	check := func(i int, x float64) {
		t.Helper()
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x) < 0x1p-1022 {
			t.Fatalf("element %d has a zero, subnormal, or non-finite component: %g", i, x)
		}
	}
	switch values := result.(type) {
	case []float64:
		for i, x := range values {
			check(i, x)
		}
	case []complex128:
		for i, x := range values {
			check(i, real(x))
			check(i, imag(x))
		}
	default:
		t.Fatalf("unsupported benchmark result type %T", result)
	}
}
