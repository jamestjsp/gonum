// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && amd64 && !safe && !noasm && !gccgo

package simdbench

import "unsafe"

func simdSlicesCompatible[T any](a, b []T) bool {
	if len(a) == 0 || len(b) == 0 {
		return true
	}
	aStart := uintptr(unsafe.Pointer(unsafe.SliceData(a)))
	bStart := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if aStart == bStart {
		return true
	}
	var value T
	size := unsafe.Sizeof(value)
	aEnd := aStart + uintptr(len(a))*size
	bEnd := bStart + uintptr(len(b))*size
	return aEnd <= bStart || bEnd <= aStart
}
