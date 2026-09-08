// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f32

func supportsAVX2SIMD() bool { return false }

const (
	nativeReductionLimitSIMD = 0
	nativeDdotShortLimitSIMD = 0
)
