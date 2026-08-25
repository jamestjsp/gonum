// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate ./single_precision.bash

package gonum

import (
	"math"

	"gonum.org/v1/gonum/internal/math32"
)

type Implementation struct{}

// [SD]gemm behavior constants. These are kept here to keep them out of the
// way during single precision code generation.
const (
	blockSize   = 64 // b x b matrix
	minParBlock = 4  // minimum number of (m,n) blocks needed to go parallel
	// minParFLOPsPerWorker is the per-worker multiply-add floor for the Dgemm
	// serial/parallel dispatch gate: the parallel path is taken only when
	// m*n*k >= minParFLOPsPerWorker * min(parBlocks, GOMAXPROCS), i.e. when
	// every worker that can be occupied receives enough work to amortize
	// goroutine fan-out overhead. Tuned via BenchmarkDgemmCrossover (see
	// dgemm_threshold_bench_test.go); with this value 100³ on 4 workers takes
	// the parallel path while shapes like 128×128×8 stay serial.
	minParFLOPsPerWorker = 1 << 17
)

// blocks returns the number of divisions of the dimension length with the given
// block size.
func blocks(dim, bsize int) int {
	return (dim + bsize - 1) / bsize
}

// dcabs1 returns |real(z)|+|imag(z)|.
func dcabs1(z complex128) float64 {
	return math.Abs(real(z)) + math.Abs(imag(z))
}

// scabs1 returns |real(z)|+|imag(z)|.
func scabs1(z complex64) float32 {
	return math32.Abs(real(z)) + math32.Abs(imag(z))
}
