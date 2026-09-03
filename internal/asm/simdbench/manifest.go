// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package simdbench holds the AMD64 assembly-to-SIMD comparison inventory.
package simdbench

// Entry describes one AMD64 assembly entry point and its Go SIMD comparison.
type Entry struct {
	Package string
	Symbol  string
	Mode    string
}

const (
	DirectSIMD    = "direct-simd"
	GatherSIMD    = "gather-simd"
	CompositeSIMD = "composite-simd"
)

// AMD64Assembly is the complete inventory of BLAS-related assembly entry
// points under internal/asm.
var AMD64Assembly = []Entry{
	{Package: "c128", Symbol: "AxpyInc", Mode: GatherSIMD},
	{Package: "c128", Symbol: "AxpyIncTo", Mode: GatherSIMD},
	{Package: "c128", Symbol: "AxpyUnitary", Mode: DirectSIMD},
	{Package: "c128", Symbol: "AxpyUnitaryTo", Mode: DirectSIMD},
	{Package: "c128", Symbol: "DotcInc", Mode: GatherSIMD},
	{Package: "c128", Symbol: "DotcUnitary", Mode: DirectSIMD},
	{Package: "c128", Symbol: "DotuInc", Mode: GatherSIMD},
	{Package: "c128", Symbol: "DotuUnitary", Mode: DirectSIMD},
	{Package: "c128", Symbol: "DscalInc", Mode: GatherSIMD},
	{Package: "c128", Symbol: "DscalUnitary", Mode: DirectSIMD},
	{Package: "c128", Symbol: "ScalInc", Mode: GatherSIMD},
	{Package: "c128", Symbol: "ScalUnitary", Mode: DirectSIMD},

	{Package: "c64", Symbol: "AxpyInc", Mode: GatherSIMD},
	{Package: "c64", Symbol: "AxpyIncTo", Mode: GatherSIMD},
	{Package: "c64", Symbol: "AxpyUnitary", Mode: DirectSIMD},
	{Package: "c64", Symbol: "AxpyUnitaryTo", Mode: DirectSIMD},
	{Package: "c64", Symbol: "DotcInc", Mode: GatherSIMD},
	{Package: "c64", Symbol: "DotcUnitary", Mode: DirectSIMD},
	{Package: "c64", Symbol: "DotuInc", Mode: GatherSIMD},
	{Package: "c64", Symbol: "DotuUnitary", Mode: DirectSIMD},

	{Package: "f32", Symbol: "AxpyInc", Mode: GatherSIMD},
	{Package: "f32", Symbol: "AxpyIncTo", Mode: GatherSIMD},
	{Package: "f32", Symbol: "AxpyUnitary", Mode: DirectSIMD},
	{Package: "f32", Symbol: "AxpyUnitaryTo", Mode: DirectSIMD},
	{Package: "f32", Symbol: "DdotInc", Mode: GatherSIMD},
	{Package: "f32", Symbol: "DdotUnitary", Mode: DirectSIMD},
	{Package: "f32", Symbol: "DotInc", Mode: GatherSIMD},
	{Package: "f32", Symbol: "DotUnitary", Mode: DirectSIMD},
	{Package: "f32", Symbol: "Ger", Mode: CompositeSIMD},
	{Package: "f32", Symbol: "Sum", Mode: DirectSIMD},

	{Package: "f64", Symbol: "Add", Mode: DirectSIMD},
	{Package: "f64", Symbol: "AddConst", Mode: DirectSIMD},
	{Package: "f64", Symbol: "AxpyInc", Mode: GatherSIMD},
	{Package: "f64", Symbol: "AxpyIncTo", Mode: GatherSIMD},
	{Package: "f64", Symbol: "AxpyUnitary", Mode: DirectSIMD},
	{Package: "f64", Symbol: "AxpyUnitaryTo", Mode: DirectSIMD},
	{Package: "f64", Symbol: "CumProd", Mode: DirectSIMD},
	{Package: "f64", Symbol: "CumSum", Mode: DirectSIMD},
	{Package: "f64", Symbol: "Div", Mode: DirectSIMD},
	{Package: "f64", Symbol: "DivTo", Mode: DirectSIMD},
	{Package: "f64", Symbol: "DotInc", Mode: GatherSIMD},
	{Package: "f64", Symbol: "DotUnitary", Mode: DirectSIMD},
	{Package: "f64", Symbol: "GemvN", Mode: CompositeSIMD},
	{Package: "f64", Symbol: "GemvT", Mode: CompositeSIMD},
	{Package: "f64", Symbol: "Ger", Mode: CompositeSIMD},
	{Package: "f64", Symbol: "L1Dist", Mode: DirectSIMD},
	{Package: "f64", Symbol: "L1Norm", Mode: DirectSIMD},
	{Package: "f64", Symbol: "L1NormInc", Mode: GatherSIMD},
	{Package: "f64", Symbol: "L2DistanceUnitary", Mode: DirectSIMD},
	{Package: "f64", Symbol: "L2NormInc", Mode: GatherSIMD},
	{Package: "f64", Symbol: "L2NormUnitary", Mode: DirectSIMD},
	{Package: "f64", Symbol: "LinfDist", Mode: DirectSIMD},
	{Package: "f64", Symbol: "ScalInc", Mode: GatherSIMD},
	{Package: "f64", Symbol: "ScalIncTo", Mode: GatherSIMD},
	{Package: "f64", Symbol: "ScalUnitary", Mode: DirectSIMD},
	{Package: "f64", Symbol: "ScalUnitaryTo", Mode: DirectSIMD},
	{Package: "f64", Symbol: "Sum", Mode: DirectSIMD},
}
