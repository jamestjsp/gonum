// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/lapack"
)

// Zlascl multiplies an m×n complex matrix by the real scalar cto/cfrom.
//
// cfrom must not be zero, and cto and cfrom must not be NaN, otherwise Zlascl
// will panic.
//
// Zlascl is an internal routine. It is exported for testing purposes.
func (impl Implementation) Zlascl(kind lapack.MatrixType, kl, ku int, cfrom, cto float64, m, n int, a []complex128, lda int) {
	switch kind {
	default:
		panic(badMatrixType)
	case 'H', 'B', 'Q', 'Z': // See zlascl.f.
		panic("not implemented")
	case lapack.General, lapack.UpperTri, lapack.LowerTri:
		if lda < max(1, n) {
			panic(badLdA)
		}
	}
	switch {
	case cfrom == 0:
		panic(zeroCFrom)
	case math.IsNaN(cfrom):
		panic(nanCFrom)
	case math.IsNaN(cto):
		panic(nanCTo)
	case m < 0:
		panic(mLT0)
	case n < 0:
		panic(nLT0)
	}

	if n == 0 || m == 0 {
		return
	}

	switch kind {
	case lapack.General, lapack.UpperTri, lapack.LowerTri:
		if len(a) < (m-1)*lda+n {
			panic(shortA)
		}
	}

	smlnum := dlamchS
	bignum := 1 / smlnum
	cfromc := cfrom
	ctoc := cto
	cfrom1 := cfromc * smlnum
	for {
		var done bool
		var mul, ctol float64
		if cfrom1 == cfromc {
			// cfromc is inf.
			mul = ctoc / cfromc
			done = true
			ctol = ctoc
		} else {
			ctol = ctoc / bignum
			if ctol == ctoc {
				// ctoc is either 0 or inf.
				mul = ctoc
				done = true
				cfromc = 1
			} else if math.Abs(cfrom1) > math.Abs(ctoc) && ctoc != 0 {
				mul = smlnum
				done = false
				cfromc = cfrom1
			} else if math.Abs(ctol) > math.Abs(cfromc) {
				mul = bignum
				done = false
				ctoc = ctol
			} else {
				mul = ctoc / cfromc
				done = true
			}
		}
		cmul := complex(mul, 0)
		switch kind {
		case lapack.General:
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					a[i*lda+j] = a[i*lda+j] * cmul
				}
			}
		case lapack.UpperTri:
			for i := 0; i < m; i++ {
				for j := i; j < n; j++ {
					a[i*lda+j] = a[i*lda+j] * cmul
				}
			}
		case lapack.LowerTri:
			for i := 0; i < m; i++ {
				for j := 0; j <= min(i, n-1); j++ {
					a[i*lda+j] = a[i*lda+j] * cmul
				}
			}
		}
		if done {
			break
		}
	}
}
