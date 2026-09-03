// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || !go1.27 || !goexperiment.simd || noasm || gccgo || safe

package f64

// DasumUnitary returns the sum of the absolute values of x.
func DasumUnitary(x []float64) float64 {
	return L1Norm(x)
}

// SwapUnitary exchanges corresponding elements of x and y.
func SwapUnitary(x, y []float64) {
	for i, v := range x {
		x[i], y[i] = y[i], v
	}
}

// RotUnitary applies a plane rotation to x and y.
func RotUnitary(x, y []float64, c, s float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = c*vx+s*vy, c*vy-s*vx
	}
}

// RotmUnitaryRescaling applies a full modified plane rotation to x and y.
func RotmUnitaryRescaling(x, y []float64, h11, h12, h21, h22 float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = float64(vx*h11)+float64(vy*h12), float64(vx*h21)+float64(vy*h22)
	}
}

// RotmUnitaryOffDiagonal applies an off-diagonal modified plane rotation.
func RotmUnitaryOffDiagonal(x, y []float64, h12, h21 float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = vx+float64(vy*h12), float64(vx*h21)+vy
	}
}

// RotmUnitaryDiagonal applies a diagonal modified plane rotation.
func RotmUnitaryDiagonal(x, y []float64, h11, h22 float64) {
	for i, vx := range x {
		vy := y[i]
		x[i], y[i] = float64(vx*h11)+vy, -vx+float64(vy*h22)
	}
}
