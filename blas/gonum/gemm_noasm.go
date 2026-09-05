// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.27 || !goexperiment.simd || safe || noasm || gccgo

package gonum

const useGEMMSIMD = false

func dgemmSerialSIMD(aTrans, bTrans bool, m, n, k int, a []float64, lda int, b []float64, ldb int, c []float64, ldc int, alpha float64) bool {
	return false
}

func sgemmSerialSIMD(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) bool {
	return false
}
