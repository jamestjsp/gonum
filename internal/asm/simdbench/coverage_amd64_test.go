// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && !safe && !noasm && !gccgo

package simdbench

import (
	"fmt"
	"math"
	"testing"

	asmc128 "gonum.org/v1/gonum/internal/asm/c128"
	asmc64 "gonum.org/v1/gonum/internal/asm/c64"
	asmf32 "gonum.org/v1/gonum/internal/asm/f32"
	asmf64 "gonum.org/v1/gonum/internal/asm/f64"
)

type kernelRun struct {
	run    func()
	result func() any
}

func TestAMD64AssemblySIMDEquivalence(t *testing.T) {
	for _, entry := range AMD64Assembly {
		entry := entry
		t.Run(entry.Package+"/"+entry.Symbol, func(t *testing.T) {
			assembly := newKernelRun(entry, 33, false)
			candidate := newKernelRun(entry, 33, true)
			assembly.run()
			candidate.run()
			if !sameKernelResult(assembly.result(), candidate.result()) {
				t.Fatalf("assembly=%v SIMD=%v", assembly.result(), candidate.result())
			}
		})
	}
}

var benchmarkSink any

func BenchmarkAMD64AssemblyVsSIMD(b *testing.B) {
	for _, entry := range AMD64Assembly {
		for _, size := range benchmarkSizes(entry) {
			for _, implementation := range []struct {
				name string
				simd bool
			}{{"assembly", false}, {"simd", true}} {
				name := fmt.Sprintf("%s/%s/n=%d/implementation=%s", entry.Package, entry.Symbol, size, implementation.name)
				b.Run(name, func(b *testing.B) {
					runner := newKernelRun(entry, size, implementation.simd)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						runner.run()
					}
					benchmarkSink = runner.result()
				})
			}
		}
	}
}

func benchmarkSizes(entry Entry) []int {
	if entry.Symbol == "Ger" || entry.Symbol == "GemvN" || entry.Symbol == "GemvT" {
		return []int{8, 64}
	}
	return []int{31, 4096}
}

func choose[T any](useSIMD bool, candidate, assembly T) T {
	if useSIMD {
		return candidate
	}
	return assembly
}

func newKernelRun(entry Entry, n int, useSIMD bool) kernelRun {
	key := entry.Package + "." + entry.Symbol
	switch key {
	case "f64.Add":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, F64AddSIMD, asmf64.Add)
		return sliceRun(func() { fn(y, x) }, y)
	case "f64.AddConst":
		x := f64Values(n, 0.2)
		fn := choose(useSIMD, F64AddConstSIMD, asmf64.AddConst)
		return sliceRun(func() { fn(0.75, x) }, x)
	case "f64.AxpyUnitary":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, F64AxpyUnitarySIMD, asmf64.AxpyUnitary)
		return sliceRun(func() { fn(0.75, x, y) }, y)
	case "f64.AxpyUnitaryTo":
		x, y, dst := f64Values(n, 0.2), f64Values(n, 0.7), make([]float64, n)
		fn := choose(useSIMD, F64AxpyUnitaryToSIMD, asmf64.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, 0.75, x, y) }, dst)
	case "f64.AxpyInc":
		x, y := f64Values(2*n, 0.2), f64Values(2*n, 0.7)
		fn := choose(useSIMD, F64AxpyIncSIMD, asmf64.AxpyInc)
		return sliceRun(func() { fn(0.75, x, y, uintptr(n), 2, 2, 0, 0) }, y)
	case "f64.AxpyIncTo":
		x, y, dst := f64Values(2*n, 0.2), f64Values(2*n, 0.7), make([]float64, 2*n)
		fn := choose(useSIMD, F64AxpyIncToSIMD, asmf64.AxpyIncTo)
		return sliceRun(func() { fn(dst, 2, 0, 0.75, x, y, uintptr(n), 2, 2, 0, 0) }, dst)
	case "f64.CumSum":
		x, dst := f64Values(n, 0.2), make([]float64, n)
		fn := choose(useSIMD, F64CumSumSIMD, asmf64.CumSum)
		return sliceRun(func() { fn(dst, x) }, dst)
	case "f64.CumProd":
		x, dst := f64Values(n, 0.98), make([]float64, n)
		fn := choose(useSIMD, F64CumProdSIMD, asmf64.CumProd)
		return sliceRun(func() { fn(dst, x) }, dst)
	case "f64.Div":
		x, y := f64Values(n, 1.2), f64Values(n, 2.1)
		fn := choose(useSIMD, F64DivSIMD, asmf64.Div)
		return sliceRun(func() { fn(y, x) }, y)
	case "f64.DivTo":
		x, y, dst := f64Values(n, 1.2), f64Values(n, 2.1), make([]float64, n)
		fn := choose(useSIMD, F64DivToSIMD, asmf64.DivTo)
		return sliceRun(func() { fn(dst, x, y) }, dst)
	case "f64.DotUnitary":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, F64DotUnitarySIMD, asmf64.DotUnitary)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.DotInc":
		x, y := f64Values(2*n, 0.2), f64Values(2*n, 0.7)
		fn := choose(useSIMD, F64DotIncSIMD, asmf64.DotInc)
		var result float64
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	case "f64.L1Norm":
		x := f64Values(n, -0.5)
		fn := choose(useSIMD, F64L1NormSIMD, asmf64.L1Norm)
		var result float64
		return scalarRun(func() { result = fn(x) }, &result)
	case "f64.L1NormInc":
		x := f64Values(2*n, -0.5)
		fn := choose(useSIMD, F64L1NormIncSIMD, asmf64.L1NormInc)
		var result float64
		return scalarRun(func() { result = fn(x, n, 2) }, &result)
	case "f64.L1Dist":
		x, y := f64Values(n, -0.5), f64Values(n, 0.9)
		fn := choose(useSIMD, F64L1DistSIMD, asmf64.L1Dist)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.LinfDist":
		x, y := f64Values(n, -0.5), f64Values(n, 0.9)
		fn := choose(useSIMD, F64LinfDistSIMD, asmf64.LinfDist)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.L2NormUnitary":
		x := f64Values(n, -0.5)
		fn := choose(useSIMD, F64L2NormUnitarySIMD, asmf64.L2NormUnitary)
		var result float64
		return scalarRun(func() { result = fn(x) }, &result)
	case "f64.L2NormInc":
		x := f64Values(2*n, -0.5)
		fn := choose(useSIMD, F64L2NormIncSIMD, asmf64.L2NormInc)
		var result float64
		return scalarRun(func() { result = fn(x, uintptr(n), 2) }, &result)
	case "f64.L2DistanceUnitary":
		x, y := f64Values(n, -0.5), f64Values(n, 0.9)
		fn := choose(useSIMD, F64L2DistanceUnitarySIMD, asmf64.L2DistanceUnitary)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f64.ScalUnitary":
		x := f64Values(n, 0.2)
		fn := choose(useSIMD, F64ScalUnitarySIMD, asmf64.ScalUnitary)
		return sliceRun(func() { fn(0.75, x) }, x)
	case "f64.ScalUnitaryTo":
		x, dst := f64Values(n, 0.2), make([]float64, n)
		fn := choose(useSIMD, F64ScalUnitaryToSIMD, asmf64.ScalUnitaryTo)
		return sliceRun(func() { fn(dst, 0.75, x) }, dst)
	case "f64.ScalInc":
		x := f64Values(2*n, 0.2)
		fn := choose(useSIMD, F64ScalIncSIMD, asmf64.ScalInc)
		return sliceRun(func() { fn(0.75, x, uintptr(n), 2) }, x)
	case "f64.ScalIncTo":
		x, dst := f64Values(2*n, 0.2), make([]float64, 2*n)
		fn := choose(useSIMD, F64ScalIncToSIMD, asmf64.ScalIncTo)
		return sliceRun(func() { fn(dst, 2, 0.75, x, uintptr(n), 2) }, dst)
	case "f64.Sum":
		x := f64Values(n, 0.2)
		fn := choose(useSIMD, F64SumSIMD, asmf64.Sum)
		var result float64
		return scalarRun(func() { result = fn(x) }, &result)
	case "f64.Ger":
		x, y := f64Values(n, 0.2), f64Values(n, 0.7)
		a := f64Values(n*n, 0.1)
		fn := choose(useSIMD, F64GerSIMD, asmf64.Ger)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, x, 1, y, 1, a, uintptr(n)) }, a)
	case "f64.GemvN":
		a, x, y := f64Values(n*n, 0.1), f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, F64GemvNSIMD, asmf64.GemvN)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, a, uintptr(n), x, 1, 0.25, y, 1) }, y)
	case "f64.GemvT":
		a, x, y := f64Values(n*n, 0.1), f64Values(n, 0.2), f64Values(n, 0.7)
		fn := choose(useSIMD, F64GemvTSIMD, asmf64.GemvT)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, a, uintptr(n), x, 1, 0.25, y, 1) }, y)
	case "f32.AxpyUnitary":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		fn := choose(useSIMD, F32AxpyUnitarySIMD, asmf32.AxpyUnitary)
		return sliceRun(func() { fn(0.75, x, y) }, y)
	case "f32.AxpyUnitaryTo":
		x, y, dst := f32Values(n, 0.2), f32Values(n, 0.7), make([]float32, n)
		fn := choose(useSIMD, F32AxpyUnitaryToSIMD, asmf32.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, 0.75, x, y) }, dst)
	case "f32.AxpyInc":
		x, y := f32Values(2*n, 0.2), f32Values(2*n, 0.7)
		fn := choose(useSIMD, F32AxpyIncSIMD, asmf32.AxpyInc)
		return sliceRun(func() { fn(0.75, x, y, uintptr(n), 2, 2, 0, 0) }, y)
	case "f32.AxpyIncTo":
		x, y, dst := f32Values(2*n, 0.2), f32Values(2*n, 0.7), make([]float32, 2*n)
		fn := choose(useSIMD, F32AxpyIncToSIMD, asmf32.AxpyIncTo)
		return sliceRun(func() { fn(dst, 2, 0, 0.75, x, y, uintptr(n), 2, 2, 0, 0) }, dst)
	case "f32.DotUnitary":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		fn := choose(useSIMD, F32DotUnitarySIMD, asmf32.DotUnitary)
		var result float32
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f32.DotInc":
		x, y := f32Values(2*n, 0.2), f32Values(2*n, 0.7)
		fn := choose(useSIMD, F32DotIncSIMD, asmf32.DotInc)
		var result float32
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	case "f32.DdotUnitary":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		fn := choose(useSIMD, F32DdotUnitarySIMD, asmf32.DdotUnitary)
		var result float64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "f32.DdotInc":
		x, y := f32Values(2*n, 0.2), f32Values(2*n, 0.7)
		fn := choose(useSIMD, F32DdotIncSIMD, asmf32.DdotInc)
		var result float64
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	case "f32.Sum":
		x := f32Values(n, 0.2)
		fn := choose(useSIMD, F32SumSIMD, asmf32.Sum)
		var result float32
		return scalarRun(func() { result = fn(x) }, &result)
	case "f32.Ger":
		x, y := f32Values(n, 0.2), f32Values(n, 0.7)
		a := f32Values(n*n, 0.1)
		fn := choose(useSIMD, F32GerSIMD, asmf32.Ger)
		return sliceRun(func() { fn(uintptr(n), uintptr(n), 0.75, x, 1, y, 1, a, uintptr(n)) }, a)
	case "c64.AxpyUnitary":
		x, y := c64Values(n, 0.2), c64Values(n, 0.7)
		fn := choose(useSIMD, C64AxpyUnitarySIMD, asmc64.AxpyUnitary)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y) }, y)
	case "c64.AxpyUnitaryTo":
		x, y, dst := c64Values(n, 0.2), c64Values(n, 0.7), make([]complex64, n)
		fn := choose(useSIMD, C64AxpyUnitaryToSIMD, asmc64.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, complex(0.75, -0.25), x, y) }, dst)
	case "c64.AxpyInc":
		x, y := c64Values(2*n, 0.2), c64Values(2*n, 0.7)
		fn := choose(useSIMD, C64AxpyIncSIMD, asmc64.AxpyInc)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y, uintptr(n), 2, 2, 0, 0) }, y)
	case "c64.AxpyIncTo":
		x, y, dst := c64Values(2*n, 0.2), c64Values(2*n, 0.7), make([]complex64, 2*n)
		fn := choose(useSIMD, C64AxpyIncToSIMD, asmc64.AxpyIncTo)
		return sliceRun(func() { fn(dst, 2, 0, complex(0.75, -0.25), x, y, uintptr(n), 2, 2, 0, 0) }, dst)
	case "c64.DotcUnitary":
		x, y := c64Values(n, 0.2), c64Values(n, 0.7)
		fn := choose(useSIMD, C64DotcUnitarySIMD, asmc64.DotcUnitary)
		var result complex64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c64.DotuUnitary":
		x, y := c64Values(n, 0.2), c64Values(n, 0.7)
		fn := choose(useSIMD, C64DotuUnitarySIMD, asmc64.DotuUnitary)
		var result complex64
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c64.DotcInc":
		x, y := c64Values(2*n, 0.2), c64Values(2*n, 0.7)
		fn := choose(useSIMD, C64DotcIncSIMD, asmc64.DotcInc)
		var result complex64
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	case "c64.DotuInc":
		x, y := c64Values(2*n, 0.2), c64Values(2*n, 0.7)
		fn := choose(useSIMD, C64DotuIncSIMD, asmc64.DotuInc)
		var result complex64
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	case "c128.AxpyUnitary":
		x, y := c128Values(n, 0.2), c128Values(n, 0.7)
		fn := choose(useSIMD, C128AxpyUnitarySIMD, asmc128.AxpyUnitary)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y) }, y)
	case "c128.AxpyUnitaryTo":
		x, y, dst := c128Values(n, 0.2), c128Values(n, 0.7), make([]complex128, n)
		fn := choose(useSIMD, C128AxpyUnitaryToSIMD, asmc128.AxpyUnitaryTo)
		return sliceRun(func() { fn(dst, complex(0.75, -0.25), x, y) }, dst)
	case "c128.AxpyInc":
		x, y := c128Values(2*n, 0.2), c128Values(2*n, 0.7)
		fn := choose(useSIMD, C128AxpyIncSIMD, asmc128.AxpyInc)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, y, uintptr(n), 2, 2, 0, 0) }, y)
	case "c128.AxpyIncTo":
		x, y, dst := c128Values(2*n, 0.2), c128Values(2*n, 0.7), make([]complex128, 2*n)
		fn := choose(useSIMD, C128AxpyIncToSIMD, asmc128.AxpyIncTo)
		return sliceRun(func() { fn(dst, 2, 0, complex(0.75, -0.25), x, y, uintptr(n), 2, 2, 0, 0) }, dst)
	case "c128.DscalUnitary":
		x := c128Values(n, 0.2)
		fn := choose(useSIMD, C128DscalUnitarySIMD, asmc128.DscalUnitary)
		return sliceRun(func() { fn(0.75, x) }, x)
	case "c128.DscalInc":
		x := c128Values(2*n, 0.2)
		fn := choose(useSIMD, C128DscalIncSIMD, asmc128.DscalInc)
		return sliceRun(func() { fn(0.75, x, uintptr(n), 2) }, x)
	case "c128.ScalUnitary":
		x := c128Values(n, 0.2)
		fn := choose(useSIMD, C128ScalUnitarySIMD, asmc128.ScalUnitary)
		return sliceRun(func() { fn(complex(0.75, -0.25), x) }, x)
	case "c128.ScalInc":
		x := c128Values(2*n, 0.2)
		fn := choose(useSIMD, C128ScalIncSIMD, asmc128.ScalInc)
		return sliceRun(func() { fn(complex(0.75, -0.25), x, uintptr(n), 2) }, x)
	case "c128.DotcUnitary":
		x, y := c128Values(n, 0.2), c128Values(n, 0.7)
		fn := choose(useSIMD, C128DotcUnitarySIMD, asmc128.DotcUnitary)
		var result complex128
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c128.DotuUnitary":
		x, y := c128Values(n, 0.2), c128Values(n, 0.7)
		fn := choose(useSIMD, C128DotuUnitarySIMD, asmc128.DotuUnitary)
		var result complex128
		return scalarRun(func() { result = fn(x, y) }, &result)
	case "c128.DotcInc":
		x, y := c128Values(2*n, 0.2), c128Values(2*n, 0.7)
		fn := choose(useSIMD, C128DotcIncSIMD, asmc128.DotcInc)
		var result complex128
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	case "c128.DotuInc":
		x, y := c128Values(2*n, 0.2), c128Values(2*n, 0.7)
		fn := choose(useSIMD, C128DotuIncSIMD, asmc128.DotuInc)
		var result complex128
		return scalarRun(func() { result = fn(x, y, uintptr(n), 2, 2, 0, 0) }, &result)
	default:
		panic("missing benchmark and equivalence runner for " + key)
	}
}

func sliceRun[T any](run func(), result []T) kernelRun {
	return kernelRun{run: run, result: func() any { return result }}
}

func scalarRun[T any](run func(), result *T) kernelRun {
	return kernelRun{run: run, result: func() any { return *result }}
}

func f64Values(n int, offset float64) []float64 {
	values := make([]float64, n)
	for i := range values {
		values[i] = offset + float64(i%17+1)/19
		if i%3 == 0 {
			values[i] = -values[i]
		}
	}
	return values
}

func f32Values(n int, offset float32) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = offset + float32(i%17+1)/19
		if i%3 == 0 {
			values[i] = -values[i]
		}
	}
	return values
}

func c64Values(n int, offset float32) []complex64 {
	realPart := f32Values(n, offset)
	imagPart := f32Values(n, offset+0.3)
	values := make([]complex64, n)
	for i := range values {
		values[i] = complex(realPart[i], imagPart[n-1-i])
	}
	return values
}

func c128Values(n int, offset float64) []complex128 {
	realPart := f64Values(n, offset)
	imagPart := f64Values(n, offset+0.3)
	values := make([]complex128, n)
	for i := range values {
		values[i] = complex(realPart[i], imagPart[n-1-i])
	}
	return values
}

func sameKernelResult(a, b any) bool {
	switch a := a.(type) {
	case float32:
		return closeFloat(float64(a), float64(b.(float32)), 2e-5)
	case float64:
		return closeFloat(a, b.(float64), 2e-12)
	case complex64:
		value := b.(complex64)
		return closeFloat(float64(real(a)), float64(real(value)), 2e-5) && closeFloat(float64(imag(a)), float64(imag(value)), 2e-5)
	case complex128:
		value := b.(complex128)
		return closeFloat(real(a), real(value), 2e-12) && closeFloat(imag(a), imag(value), 2e-12)
	case []float32:
		value := b.([]float32)
		for i := range a {
			if !closeFloat(float64(a[i]), float64(value[i]), 2e-5) {
				return false
			}
		}
		return true
	case []float64:
		value := b.([]float64)
		for i := range a {
			if !closeFloat(a[i], value[i], 2e-12) {
				return false
			}
		}
		return true
	case []complex64:
		value := b.([]complex64)
		for i := range a {
			if !sameKernelResult(a[i], value[i]) {
				return false
			}
		}
		return true
	case []complex128:
		value := b.([]complex128)
		for i := range a {
			if !sameKernelResult(a[i], value[i]) {
				return false
			}
		}
		return true
	default:
		panic(fmt.Sprintf("unsupported result type %T", a))
	}
}

func closeFloat(a, b, tolerance float64) bool {
	if a == b {
		return true
	}
	return math.Abs(a-b) <= tolerance*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}
