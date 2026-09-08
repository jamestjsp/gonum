// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 && go1.27) || (!amd64 && !arm64) || noasm || gccgo || safe
// +build amd64,go1.27 !amd64,!arm64 noasm gccgo safe

package math32

import (
	"math"
)

// Sqrt returns the square root of x.
//
// Special cases are:
//
//	Sqrt(+Inf) = +Inf
//	Sqrt(±0) = ±0
//	Sqrt(x < 0) = NaN
//	Sqrt(NaN) = NaN
func Sqrt(x float32) float32 {
	// On current AMD64 Go toolchains this expression lowers to a scalar
	// float32 square root, avoiding conversions and an assembly ABI call.
	return float32(math.Sqrt(float64(x)))
}
