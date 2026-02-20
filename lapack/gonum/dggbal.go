// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"

	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/gonum/lapack"
)

// Dggbal balances a pair of general real matrices (A,B). Balancing involves:
//  1. Permuting rows and columns to isolate eigenvalues
//  2. Applying diagonal similarity transformations to make rows and columns
//     as close in norm as possible
//
// This may improve the accuracy of computed eigenvalues and eigenvectors
// in the generalized eigenvalue problem A*x = lambda*B*x.
//
// job specifies the operations to perform:
//   - lapack.BalanceNone: set lscale[i]=rscale[i]=1 for all i, return ilo=0, ihi=n-1
//   - lapack.Permute: permute only
//   - lapack.Scale: scale only
//   - lapack.PermuteScale: both permute and scale
//
// On return, A and B are overwritten by the balanced matrices.
//
// ilo and ihi mark the starting and ending columns of the balanced submatrix.
// For permuting, A[i,j]=0 and B[i,j]=0 for i>j and j in {0,...,ilo-1,ihi+1,...,n-1}.
//
// lscale and rscale contain the left and right scaling factors:
//   - For j in {0,...,ilo-1,ihi+1,...,n-1}: permutation indices
//   - For j in {ilo,...,ihi}: scaling factors
//
// lscale, rscale must have length n. work must have length at least max(1,6*n).
//
// Dggbal is an internal routine. It is exported for testing purposes.
func (impl Implementation) Dggbal(job lapack.BalanceJob, n int, a []float64, lda int, b []float64, ldb int, lscale, rscale, work []float64) (ilo, ihi int) {
	switch {
	case job != lapack.BalanceNone && job != lapack.Permute && job != lapack.Scale && job != lapack.PermuteScale:
		panic(badBalanceJob)
	case n < 0:
		panic(nLT0)
	case lda < max(1, n):
		panic(badLdA)
	case ldb < max(1, n):
		panic(badLdB)
	}

	ilo = 0
	ihi = n - 1

	if n == 0 {
		return ilo, ihi
	}

	switch {
	case len(lscale) < n:
		panic(shortLscale)
	case len(rscale) < n:
		panic(shortRscale)
	}

	if job == lapack.BalanceNone {
		for i := 0; i < n; i++ {
			lscale[i] = 1
			rscale[i] = 1
		}
		return ilo, ihi
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(shortA)
	case len(b) < (n-1)*ldb+n:
		panic(shortB)
	case len(work) < max(1, 6*n):
		panic(shortWork)
	}

	bi := blas64.Implementation()

	if job != lapack.Scale {
		// Permute rows/columns to isolate eigenvalues.

		// Find rows/columns containing only zeros in one matrix.
		// These correspond to infinite eigenvalues.

		// Search for rows isolating an eigenvalue and push them down.
		ihi = n - 1
		for {
			if ihi == 0 {
				rscale[0] = 1
				lscale[0] = 1
				goto done
			}
			found := false
		rowLoop:
			for i := ihi; i >= 0; i-- {
				foundNonzero := false
				for j := 0; j <= ihi; j++ {
					if i == j {
						continue
					}
					if a[i*lda+j] != 0 || b[i*ldb+j] != 0 {
						foundNonzero = true
						break
					}
				}
				if !foundNonzero {
					// Row i has only zero off-diagonal elements.
					rscale[ihi] = float64(i)
					lscale[ihi] = float64(i)
					if i != ihi {
						bi.Dswap(ihi+1, a[i:], lda, a[ihi:], lda)
						bi.Dswap(n, a[i*lda:], 1, a[ihi*lda:], 1)
						bi.Dswap(ihi+1, b[i:], ldb, b[ihi:], ldb)
						bi.Dswap(n, b[i*ldb:], 1, b[ihi*ldb:], 1)
					}
					found = true
					ihi--
					break rowLoop
				}
			}
			if !found {
				break
			}
		}

		// Search for columns isolating an eigenvalue and push them left.
		ilo = 0
		for {
			found := false
		colLoop:
			for j := ilo; j <= ihi; j++ {
				foundNonzero := false
				for i := ilo; i <= ihi; i++ {
					if i == j {
						continue
					}
					if a[i*lda+j] != 0 || b[i*ldb+j] != 0 {
						foundNonzero = true
						break
					}
				}
				if !foundNonzero {
					// Column j has only zero off-diagonal elements.
					rscale[ilo] = float64(j)
					lscale[ilo] = float64(j)
					if j != ilo {
						bi.Dswap(ihi+1, a[j:], lda, a[ilo:], lda)
						bi.Dswap(n-ilo, a[j*lda+ilo:], 1, a[ilo*lda+ilo:], 1)
						bi.Dswap(ihi+1, b[j:], ldb, b[ilo:], ldb)
						bi.Dswap(n-ilo, b[j*ldb+ilo:], 1, b[ilo*ldb+ilo:], 1)
					}
					found = true
					ilo++
					break colLoop
				}
			}
			if !found {
				break
			}
		}
	}

done:
	// Initialize scaling factors.
	for i := ilo; i <= ihi; i++ {
		lscale[i] = 1
		rscale[i] = 1
	}

	if job == lapack.Permute || ilo == ihi {
		return ilo, ihi
	}

	// Balance the submatrix in rows ilo to ihi.
	nr := ihi - ilo + 1

	// Compute norms using work arrays.
	// work[0:n]   = row norms of A
	// work[n:2n]  = row norms of B
	// work[2n:3n] = column norms of A
	// work[3n:4n] = column norms of B
	// work[4n:5n] = row scaling lscale
	// work[5n:6n] = column scaling rscale

	for i := ilo; i <= ihi; i++ {
		work[i] = 0
		work[n+i] = 0
		work[2*n+i] = 0
		work[3*n+i] = 0
	}

	for i := ilo; i <= ihi; i++ {
		for j := ilo; j <= ihi; j++ {
			work[i] += math.Abs(a[i*lda+j])
			work[n+i] += math.Abs(b[i*ldb+j])
			work[2*n+j] += math.Abs(a[i*lda+j])
			work[3*n+j] += math.Abs(b[i*ldb+j])
		}
	}

	// Balance with iterative scaling.
	const (
		sclfac = 10.0
		factor = 0.95
	)
	sfmin := dlamchS
	sfmax := 1 / sfmin

	conv := false
	maxIter := 5 * nr
	for iter := 0; !conv && iter < maxIter; iter++ {
		conv = true
		for i := ilo; i <= ihi; i++ {
			// Compute row and column norms.
			rowA := work[i]
			rowB := work[n+i]
			colA := work[2*n+i]
			colB := work[3*n+i]

			if rowA+rowB == 0 || colA+colB == 0 {
				continue
			}

			// Row norm.
			rowNorm := rowA + rowB
			// Column norm.
			colNorm := colA + colB

			// Compute geometric mean of row and column norms.
			target := math.Sqrt(rowNorm * colNorm)

			// Compute scaling factors.
			f := 1.0
			g := rowNorm / sclfac

			// Scale up row.
			for colNorm < g && math.Max(f, math.Max(colA, colB)) < sfmax/sclfac {
				f *= sclfac
				colA *= sclfac
				colB *= sclfac
				colNorm = colA + colB
				g /= sclfac
			}

			// Scale down row.
			g = colNorm / sclfac
			for rowNorm < g && math.Min(rowA, rowB) > sfmin*sclfac {
				f /= sclfac
				rowA /= sclfac
				rowB /= sclfac
				rowNorm = rowA + rowB
				g /= sclfac
			}

			// Check for convergence.
			if colNorm+rowNorm >= factor*target {
				continue
			}

			// Apply scaling.
			if f < 1 && lscale[i] < 1 && f*lscale[i] <= sfmin {
				continue
			}
			if f > 1 && lscale[i] > 1 && lscale[i] >= sfmax/f {
				continue
			}

			// Scale row.
			lscale[i] *= f
			work[i] = rowA
			work[n+i] = rowB
			bi.Dscal(nr, f, a[i*lda+ilo:], 1)
			bi.Dscal(nr, f, b[i*ldb+ilo:], 1)

			// Inverse scale column.
			rscale[i] *= f
			work[2*n+i] = colA
			work[3*n+i] = colB
			bi.Dscal(nr, 1/f, a[ilo*lda+i:], lda)
			bi.Dscal(nr, 1/f, b[ilo*ldb+i:], ldb)

			conv = false
		}
	}

	return ilo, ihi
}

const (
	shortLscale = "lapack: insufficient length of lscale"
	shortRscale = "lapack: insufficient length of rscale"
)
