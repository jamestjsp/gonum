// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

func dotShortHardwareSIMD(x, y []float32) float32 { return dotUnitaryOriginalSIMD(x, y) }
func sumShortHardwareSIMD(x []float32) float32    { return sumOriginalSIMD(x) }
