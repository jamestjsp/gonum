// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"testing"
)

// Preserve the original entry routing as an independent admission oracle.
// Both native arithmetic leaves and the checked hardware helpers are unchanged.
func gemvTOriginalEntrySIMD(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr) {
	if n >= 4 && n <= 16 && incX == 1 && incY == 1 {
		if n == 8 {
			if gemvTEightHardwareSIMD(m, alpha, a, lda, x, beta, y) {
				return
			}
		} else if gemvTSmallHardwareSIMD(m, n, alpha, a, lda, x, beta, y) {
			return
		}
	}
	gemvTPortableSIMD(m, n, alpha, a, lda, x, incX, beta, y, incY)
}

type gemvTEntryParityCase struct {
	m, n, incX, incY int
	alpha, beta      float64
	alias, invalid   string
	exceptional      bool
}

func (c gemvTEntryParityCase) run(candidate bool) (data []float64, panicked bool) {
	span := func(n, inc int) int {
		if n == 0 {
			return 0
		}
		if inc < 0 {
			inc = -inc
		}
		return (n-1)*inc + 1
	}
	xLen, yLen := span(c.m, c.incX), span(c.n, c.incY)
	aLen := 0
	if c.m > 0 && c.n > 0 {
		aLen = (c.m-1)*(c.n+3) + c.n
	}
	data = make([]float64, aLen+xLen+yLen+16)
	special := []float64{0, math.Copysign(0, -1), 1, -1, math.SmallestNonzeroFloat64, .75 * math.MaxFloat64, -.75 * math.MaxFloat64, math.Inf(1), math.Float64frombits(0x7ff8000000000042)}
	for i := range data {
		data[i] = float64(i%19-9) * .125
		if c.exceptional {
			data[i] = special[i%len(special)]
		}
	}
	xStart, yStart := aLen+6, aLen+xLen+10
	switch c.alias {
	case "xy":
		yStart = xStart + 1
	case "ay":
		yStart = 3
	case "all":
		xStart, yStart = 3, 4
	}
	a := data[2 : 2+aLen : 2+aLen]
	x := data[xStart : xStart+xLen : xStart+xLen]
	y := data[yStart : yStart+yLen : yStart+yLen]
	lda := uintptr(c.n + 3)
	switch c.invalid {
	case "x":
		x = x[: len(x)-1 : len(x)-1]
	case "y":
		y = y[: len(y)-1 : len(y)-1]
	case "a":
		a = a[: len(a)-1 : len(a)-1]
	case "rows":
		lda = uintptr(c.n - 1)
	case "huge_lda":
		lda = ^uintptr(0)
	}
	panicked = true
	defer func() { _ = recover() }()
	fn := gemvTOriginalEntrySIMD
	if candidate {
		fn = GemvTSIMD
	}
	fn(uintptr(c.m), uintptr(c.n), c.alpha, a, lda, x, uintptr(c.incX), c.beta, y, uintptr(c.incY))
	panicked = false
	return data, panicked
}

func TestSIMDGemvTPublicEntryParity(t *testing.T) {
	var cases []gemvTEntryParityCase
	for _, m := range []int{0, 1, 3, 4, 7, 8, 65} {
		for _, n := range []int{0, 3, 4, 5, 7, 8, 9, 15, 16, 17} {
			for _, inc := range [][2]int{{1, 1}, {2, 3}, {-3, 1}, {1, -2}, {0, 1}, {1, 0}} {
				cases = append(cases, gemvTEntryParityCase{m: m, n: n, incX: inc[0], incY: inc[1], alpha: .5, beta: -.25})
			}
		}
	}
	for _, n := range []int{4, 7, 8, 15, 16} {
		for _, alias := range []string{"", "xy", "ay", "all"} {
			for _, beta := range []float64{0, math.Copysign(0, -1), .5, math.Inf(1)} {
				for _, exceptional := range []bool{false, true} {
					for _, alpha := range []float64{0, math.Copysign(0, -1), -.75, math.Inf(1)} {
						cases = append(cases, gemvTEntryParityCase{m: 8, n: n, incX: 1, incY: 1, alpha: alpha, beta: beta, alias: alias, exceptional: exceptional})
					}
				}
			}
		}
		for _, invalid := range []string{"x", "y", "a", "rows"} {
			cases = append(cases, gemvTEntryParityCase{m: 8, n: n, incX: 1, incY: 1, alpha: .5, beta: .25, invalid: invalid})
		}
		for _, m := range []int{1, 2} {
			cases = append(cases, gemvTEntryParityCase{m: m, n: n, incX: 1, incY: 1, alpha: .5, beta: 0, invalid: "huge_lda"})
		}
	}
	for i, c := range cases {
		t.Run(fmt.Sprintf("case=%d/m=%d/n=%d/inc=%d,%d/alias=%s/invalid=%s", i, c.m, c.n, c.incX, c.incY, c.alias, c.invalid), func(t *testing.T) {
			want, wantPanic := c.run(false)
			got, gotPanic := c.run(true)
			if gotPanic != wantPanic {
				t.Fatalf("panic state: got%t want%t", gotPanic, wantPanic)
			}
			// Include guard sentinels, inputs, row padding and any mutations
			// made before a checked invalid-span panic. Payload bits matter.
			for j := range want {
				if math.Float64bits(got[j]) != math.Float64bits(want[j]) {
					t.Fatalf("storage[%d]: got%016x want%016x", j, math.Float64bits(got[j]), math.Float64bits(want[j]))
				}
			}
		})
	}
}
