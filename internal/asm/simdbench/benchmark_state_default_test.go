// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && (!simdbenchclean || !amd64) && !safe && !noasm && !gccgo

package simdbench

func clearBenchmarkAVXState() {}
