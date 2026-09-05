// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

import "simd"

// Preserve the complete original accumulation, lane reduction and scalar-tail
// order for exceptional retries. Replaying only the final lane reduction is
// insufficient when a changed finite reduction overflows upon adding a tail.
func dotUnitaryOriginalSIMD(x, y []float32) float32 {
	acc := simd.BroadcastFloat32s(0)
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	y = y[:len(x):len(x)]
	for len(x) >= 4*width {
		xblock, yblock := x[:4*width], y[:4*width]
		acc = simd.LoadFloat32s(xblock[:width]).Mul(simd.LoadFloat32s(yblock[:width])).Add(acc)
		acc1 = simd.LoadFloat32s(xblock[width : 2*width]).Mul(simd.LoadFloat32s(yblock[width : 2*width])).Add(acc1)
		acc2 = simd.LoadFloat32s(xblock[2*width : 3*width]).Mul(simd.LoadFloat32s(yblock[2*width : 3*width])).Add(acc2)
		acc3 = simd.LoadFloat32s(xblock[3*width : 4*width]).Mul(simd.LoadFloat32s(yblock[3*width : 4*width])).Add(acc3)
		x, y = x[4*width:], y[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat32s(x[:width]).Mul(simd.LoadFloat32s(y[:width])).Add(acc)
		x, y = x[width:], y[width:]
	}
	sum := reduceF32Portable(acc)
	for i, value := range x {
		sum += value * y[i]
	}
	return sum
}

func sumOriginalSIMD(x []float32) float32 {
	acc := simd.BroadcastFloat32s(0)
	acc1, acc2, acc3 := acc, acc, acc
	width := acc.Len()
	for len(x) >= 4*width {
		acc = simd.LoadFloat32s(x[:width]).Add(acc)
		acc1 = simd.LoadFloat32s(x[width : 2*width]).Add(acc1)
		acc2 = simd.LoadFloat32s(x[2*width : 3*width]).Add(acc2)
		acc3 = simd.LoadFloat32s(x[3*width : 4*width]).Add(acc3)
		x = x[4*width:]
	}
	acc = acc.Add(acc1).Add(acc2.Add(acc3))
	for len(x) >= width {
		acc = simd.LoadFloat32s(x[:width]).Add(acc)
		x = x[width:]
	}
	sum := reduceF32Portable(acc)
	for _, value := range x {
		sum += value
	}
	return sum
}
