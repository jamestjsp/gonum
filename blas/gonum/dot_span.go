// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "math/bits"

// checkedDotStart returns the initial index of a dot-product vector if all n
// elements fit within length. The caller must first check n > 0 and inc != 0.
func checkedDotStart(n, inc, length int) (start int, ok bool) {
	stride := uint(inc)
	if inc < 0 {
		// Unsigned negation also gives the magnitude of MinInt.
		stride = -stride
	}
	hi, span := bits.Mul(uint(n-1), stride)
	if hi != 0 || span >= uint(length) {
		return 0, false
	}
	if inc < 0 {
		// span < length proves that the conversion cannot overflow int.
		return int(span), true
	}
	return 0, true
}
