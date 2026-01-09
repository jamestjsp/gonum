// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "gonum.org/v1/gonum/blas"

// Dtgexc reorders the generalized Schur decomposition of a real matrix pair
// (A,B), so that the diagonal block of (A,B) at row ifst is moved to row ilst.
//
// The matrix pair (A,B) is in generalized Schur form: A is quasi-upper
// triangular (block upper triangular with 1x1 and 2x2 blocks) and B is upper
// triangular. After reordering, the matrices are updated:
//
//	A := Q^T * A * Z
//	B := Q^T * B * Z
//
// where Q and Z are orthogonal matrices.
//
// If wantq is true, Q is updated. If wantz is true, Z is updated.
//
// ifst and ilst are 0-indexed positions of the diagonal blocks.
//
// work must have length at least lwork. If lwork is -1, a workspace query is
// performed and the optimal size is returned in work[0].
//
// The return values ifstOut and ilstOut give the final positions.
// ok is false if the swap failed (ill-conditioned problem).
//
// Dtgexc is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
	q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, lwork int) (ifstOut, ilstOut int, ok bool) {

	switch {
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	case ldq < 1, wantq && ldq < n:
		panic(badLdQ)
	case ldz < 1, wantz && ldz < n:
		panic(badLdZ)
	case lwork < max(1, 4*n+16) && lwork != -1:
		panic(badLWork)
	}

	// Workspace query.
	if lwork == -1 {
		work[0] = float64(4*n + 16)
		return ifst, ilst, true
	}

	if n == 0 {
		return ifst, ilst, true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case wantq && len(q) < (n-1)*ldq+n:
		panic(shortQ)
	case wantz && len(z) < (n-1)*ldz+n:
		panic(shortZ)
	case len(work) < lwork:
		panic(shortWork)
	}

	switch {
	case ifst < 0 || ifst >= n:
		panic(badIfst)
	case ilst < 0 || ilst >= n:
		panic(badIlst)
	}

	ifstOut = ifst
	ilstOut = ilst
	ok = true

	if n == 1 {
		return ifstOut, ilstOut, ok
	}

	// Initialize Q and Z to identity if needed.
	if wantq {
		impl.Dlaset(blas.All, n, n, 0, 1, q, ldq)
	}
	if wantz {
		impl.Dlaset(blas.All, n, n, 0, 1, z, ldz)
	}

	here := ifstOut

	if ilstOut > ifstOut {
		// Move block down.
		for here < ilstOut {
			// Determine the size of the current block.
			nbf := 1
			if here < n-1 && a[(here+1)*lda+here] != 0 {
				nbf = 2
			}
			// Determine the size of the next block.
			nbl := 1
			if here+nbf < n-1 && a[(here+nbf+1)*lda+here+nbf] != 0 {
				nbl = 2
			}

			// Swap the blocks.
			swapOk := impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here, nbf, nbl, work, lwork)
			if !swapOk {
				ilstOut = here
				ok = false
				return ifstOut, ilstOut, ok
			}

			here += nbl

			if nbl == 2 && here+1 < n && a[(here+1)*lda+here] == 0 {
				// 2x2 block became two 1x1 blocks; adjust position.
			}
		}
	} else if ilstOut < ifstOut {
		// Move block up.
		for here > ilstOut {
			// Determine the size of the current block.
			nbf := 1
			if here > 0 && a[here*lda+here-1] != 0 {
				nbf = 2
				here--
			}
			// Determine the size of the previous block.
			nbl := 1
			if here >= 2 && a[(here-1)*lda+here-2] != 0 {
				nbl = 2
			}

			// Ensure we don't go below 0.
			if here < nbl {
				break
			}

			// Swap the blocks.
			swapOk := impl.Dtgex2(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, here-nbl, nbl, nbf, work, lwork)
			if !swapOk {
				ilstOut = here
				ok = false
				return ifstOut, ilstOut, ok
			}

			here -= nbl
		}
	}

	ifstOut = here
	ilstOut = here
	return ifstOut, ilstOut, ok
}
