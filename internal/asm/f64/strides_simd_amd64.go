// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

// Sparse scalar loads and stores outperform stack packing on the measured
// amd64 backends. Keep other architectures on their portable vector path.
const scalarStridesSIMD = true

const scalarPrefixesSIMD = true
