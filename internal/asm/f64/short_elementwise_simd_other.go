// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func shortElementwiseHardwareSIMD(n int) bool                         { return false }
func linfShortHardwareSIMD(x, y []float64) (float64, bool)            { return 0, false }
func scalShortHardwareSIMD(dst []float64, alpha float64, x []float64) {}

func shortScalHardwareSIMD(n int) bool { return false }

func scalToHardwareSIMD(n int) bool { return false }
