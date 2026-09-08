// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

func L1NormSIMD(x []float64) float64    { return l1NormPortableSIMD(x) }
func L1DistSIMD(x, y []float64) float64 { return l1DistPortableSIMD(x, y) }
