// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

func axpyTailSIMD(dst []float32, alpha float32, x, y []float32) int       { return 0 }
func gerTailSIMD(a0, a1, a2, a3, y []float32, s0, s1, s2, s3 float32) int { return 0 }
