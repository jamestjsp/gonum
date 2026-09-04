// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"simd"
	"slices"
	"testing"
)

func TestPortableSIMDZeroLength(t *testing.T) {
	offset := ^uintptr(0)
	AxpyIncSIMD(2, nil, nil, 0, 1, 1, offset, offset)
	AxpyIncToSIMD(nil, 1, offset, 2, nil, nil, 0, 1, 1, offset, offset)
	ScalIncSIMD(2, nil, 0, 1)
	ScalIncToSIMD(nil, 1, 2, nil, 0, 1)
	for name, got := range map[string]float64{
		"DotInc":     DotIncSIMD(nil, nil, 0, 1, 1, offset, offset),
		"DotUnitary": DotUnitarySIMD(nil, nil),
		"Sum":        SumSIMD(nil),
		"L1NormInc":  L1NormIncSIMD(nil, 0, 1),
		"L2NormInc":  L2NormIncSIMD(nil, 0, 1),
	} {
		if got != 0 {
			t.Errorf("%s: got %g, want zero", name, got)
		}
	}
}

func TestPortableSIMDIncrements(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 65, 257} {
		for _, inc := range []int{1, 2, -1} {
			t.Run(fmt.Sprintf("n=%d/inc=%d", n, inc), func(t *testing.T) {
				x, y := portableSIMDValues(2*n+9), portableSIMDValues(2*n+11)
				ix, iy, idst := 2, 3, 4
				if inc < 0 && n > 0 {
					ix += n - 1
					iy += n - 1
					idst += n - 1
				}
				wantY := slices.Clone(y)
				dst := portableSIMDValues(2*n + 13)
				wantDst := slices.Clone(dst)
				var dot float64
				for i := range n {
					xv, yv := x[ix+i*inc], y[iy+i*inc]
					wantY[iy+i*inc] = -0.75*xv + yv
					wantDst[idst+i*inc] = -0.75*xv + yv
					dot += xv * yv
				}
				portableSIMDCheck(t, "DotInc", DotIncSIMD(x, y, uintptr(n), uintptr(inc), uintptr(inc), uintptr(ix), uintptr(iy)), dot)
				AxpyIncToSIMD(dst, uintptr(inc), uintptr(idst), -0.75, x, y, uintptr(n), uintptr(inc), uintptr(inc), uintptr(ix), uintptr(iy))
				portableSIMDCheckSlice(t, "AxpyIncTo", dst, wantDst)
				AxpyIncSIMD(-0.75, x, y, uintptr(n), uintptr(inc), uintptr(inc), uintptr(ix), uintptr(iy))
				portableSIMDCheckSlice(t, "AxpyInc", y, wantY)
				if inc < 0 {
					return
				}
				var l1, l2 float64
				wantX := slices.Clone(x)
				wantDst = slices.Clone(dst)
				for i := range n {
					v := x[2+i*inc]
					l1 += math.Abs(v)
					l2 = math.Hypot(l2, v)
					wantX[2+i*inc] = -0.75 * v
					wantDst[4+i*inc] = -0.75 * v
				}
				portableSIMDCheck(t, "L1NormInc", L1NormIncSIMD(x[2:], n, inc), l1)
				portableSIMDCheck(t, "L2NormInc", L2NormIncSIMD(x[2:], uintptr(n), uintptr(inc)), l2)
				ScalIncToSIMD(dst[4:], uintptr(inc), -0.75, x[2:], uintptr(n), uintptr(inc))
				portableSIMDCheckSlice(t, "ScalIncTo", dst, wantDst)
				ScalIncSIMD(-0.75, x[2:], uintptr(n), uintptr(inc))
				portableSIMDCheckSlice(t, "ScalInc", x, wantX)
			})
		}
		x, y := portableSIMDValues(n), portableSIMDValues(n + 1)[1:]
		var sum, dot float64
		for i, v := range x {
			sum += v
			dot += v * y[i]
		}
		portableSIMDCheck(t, fmt.Sprintf("Sum/n=%d", n), SumSIMD(x), sum)
		portableSIMDCheck(t, fmt.Sprintf("DotUnitary/n=%d", n), DotUnitarySIMD(x, y), dot)
	}
}

func TestPortableSIMDStridedBits(t *testing.T) {
	values := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, 1, -2, math.Inf(1), math.Inf(-1), math.Float64frombits(0x7ff8000000000123)}
	n := 2*simd.BroadcastFloat64s(0).Len() + len(values) + 1
	for _, backward := range []bool{false, true} {
		x, y, dst := portableSIMDValues(2*n+3), portableSIMDValues(3*n+4), portableSIMDValues(2*n+5)
		ix, iy, idst := uintptr(1), uintptr(2), uintptr(3)
		incX, incY, incDst := uintptr(2), uintptr(3), uintptr(2)
		if backward {
			ix += uintptr(n-1) * incX
			iy += uintptr(n-1) * incY
			idst += uintptr(n-1) * incDst
			incX, incY, incDst = -incX, -incY, -incDst
		}
		for i, jx, jy := 0, ix, iy; i < n; i++ {
			x[jx], y[jy] = values[i%len(values)], values[(i+2)%len(values)]
			jx, jy = jx+incX, jy+incY
		}
		wantDst, wantY := slices.Clone(dst), slices.Clone(y)
		var dot float64
		for i, jx, jy, jd := 0, ix, iy, idst; i < n; i++ {
			wantDst[jd] = 0.5*x[jx] + y[jy]
			wantY[jy] += 0.5 * x[jx]
			dot += x[jx] * y[jy]
			jx, jy, jd = jx+incX, jy+incY, jd+incDst
		}
		gotDot := DotIncSIMD(x, y, uintptr(n), incX, incY, ix, iy)
		portableSIMDCheckBits(t, fmt.Sprintf("DotInc/backward=%t", backward), []float64{gotDot}, []float64{dot})
		AxpyIncToSIMD(dst, incDst, idst, 0.5, x, y, uintptr(n), incX, incY, ix, iy)
		portableSIMDCheckBits(t, fmt.Sprintf("AxpyIncTo/backward=%t", backward), dst, wantDst)
		AxpyIncSIMD(0.5, x, y, uintptr(n), incX, incY, ix, iy)
		portableSIMDCheckBits(t, fmt.Sprintf("AxpyInc/backward=%t", backward), y, wantY)
	}
	x, dst := portableSIMDValues(2*n+3), portableSIMDValues(3*n+4)
	var l1 float64
	for i := range n {
		x[2*i] = values[i%len(values)]
		l1 += math.Abs(x[2*i])
	}
	wantX, wantDst := slices.Clone(x), slices.Clone(dst)
	for i := range n {
		wantDst[3*i] = x[2*i]
	}
	portableSIMDCheckBits(t, "L1NormInc", []float64{L1NormIncSIMD(x, n, 2)}, []float64{l1})
	ScalIncToSIMD(dst, 3, 1, x, uintptr(n), 2)
	portableSIMDCheckBits(t, "ScalIncTo", dst, wantDst)
	ScalIncSIMD(1, x, uintptr(n), 2)
	portableSIMDCheckBits(t, "ScalInc", x, wantX)
}

func portableSIMDCheckBits(t *testing.T, name string, got, want []float64) {
	t.Helper()
	for i, v := range got {
		if math.Float64bits(v) != math.Float64bits(want[i]) && !(math.IsNaN(v) && math.IsNaN(want[i])) {
			t.Fatalf("%s[%d]: got %g (%#x), want %g (%#x)", name, i, v, math.Float64bits(v), want[i], math.Float64bits(want[i]))
		}
	}
}

func TestPortableSIMDPrefix(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 65} {
		for _, product := range []bool{false, true} {
			for _, layout := range []struct {
				name     string
				src, dst int
			}{
				{"separate", 2, n + 5},
				{"in-place", 2, 2},
				{"destination ahead", 2, 3},
				{"destination behind", 3, 2},
			} {
				t.Run(fmt.Sprintf("n=%d/product=%t/%s", n, product, layout.name), func(t *testing.T) {
					got := make([]float64, 2*n+8)
					for i := range got {
						got[i] = -12345
					}
					for i := range n {
						value := float64(i%7-3) / 8
						if product {
							value = 1 + float64(i%7-3)/100
						}
						if i == 0 {
							value = 0x1p-65
							if product {
								value = 1
							}
						}
						got[layout.src+i] = value
					}
					want := slices.Clone(got)
					carry := 0.0
					fn := CumSumSIMD
					if product {
						carry, fn = 1, CumProdSIMD
					}
					for i := range n {
						if product {
							carry *= want[layout.src+i]
						} else {
							carry += want[layout.src+i]
						}
						want[layout.dst+i] = carry
					}
					result := fn(got[layout.dst:layout.dst+n], got[layout.src:layout.src+n])
					if len(result) != n {
						t.Fatalf("returned length %d, want %d", len(result), n)
					}
					portableSIMDCheckSlice(t, "prefix backing slice", got, want)
				})
			}
		}
	}
}

func TestPortableSIMDMatrix(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {3, 17}, {17, 3}, {8, 8}, {33, 65}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, incs := range [][2]int{{1, 1}, {2, 3}, {-1, 1}, {1, -2}, {-2, -3}} {
			incX, incY := incs[0], incs[1]
			t.Run(fmt.Sprintf("m=%d/n=%d/incX=%d/incY=%d", m, n, incX, incY), func(t *testing.T) {
				a := portableSIMDValues(m*lda + 2)
				x, ix := portableSIMDVector(m, incX)
				y, iy := portableSIMDVector(n, incY)
				want := slices.Clone(a)
				for i := range m {
					for j := range n {
						want[i*lda+j] += -0.75 * x[ix+i*incX] * y[iy+j*incY]
					}
				}
				GerSIMD(uintptr(m), uintptr(n), -0.75, x, uintptr(incX), y, uintptr(incY), a, uintptr(lda))
				portableSIMDCheckSlice(t, "Ger", a, want)
				for _, trans := range []bool{false, true} {
					rows, cols := m, n
					if trans {
						rows, cols = n, m
					}
					for _, beta := range []float64{0, 0.25, -1} {
						x, ix := portableSIMDVector(cols, incX)
						y, iy := portableSIMDVector(rows, incY)
						if beta == 0 {
							for i := range rows {
								y[iy+i*incY] = math.NaN()
							}
						}
						want := slices.Clone(y)
						for i := range rows {
							var dot float64
							for j := range cols {
								index := i*lda + j
								if trans {
									index = j*lda + i
								}
								dot += a[index] * x[ix+j*incX]
							}
							value := -0.75 * dot
							if beta != 0 {
								value += beta * y[iy+i*incY]
							}
							want[iy+i*incY] = value
						}
						fn := GemvNSIMD
						if trans {
							fn = GemvTSIMD
						}
						fn(uintptr(m), uintptr(n), -0.75, a, uintptr(lda), x, uintptr(incX), beta, y, uintptr(incY))
						portableSIMDCheckSlice(t, fmt.Sprintf("Gemv/trans=%t/beta=%g", trans, beta), y, want)
					}
				}
			})
		}
	}
}

func portableSIMDValues(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64((i*7)%29-14) / 13
	}
	return x
}

func portableSIMDVector(n, inc int) ([]float64, int) {
	start := 0
	if inc < 0 {
		inc = -inc
		start = (n - 1) * inc
	}
	return portableSIMDValues((n-1)*inc + 4), start
}

func portableSIMDCheckSlice(t *testing.T, name string, got, want []float64) {
	t.Helper()
	for i, v := range got {
		portableSIMDCheck(t, fmt.Sprintf("%s[%d]", name, i), v, want[i])
	}
}

func portableSIMDCheck(t *testing.T, name string, got, want float64) {
	t.Helper()
	if got == want {
		return
	}
	if math.IsNaN(got) || math.Abs(got-want) > 2e-12*math.Max(1, math.Abs(want)) {
		t.Fatalf("%s: got %.17g, want %.17g", name, got, want)
	}
}
