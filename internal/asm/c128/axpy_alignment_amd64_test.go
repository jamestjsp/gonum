// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && !safe && !noasm && !gccgo

package c128

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"testing"
	"unsafe"
)

// The arrays are ordinary Go storage. Their starts differ by 16*128+8
// bytes, so they provide both valid complex128 alignment classes regardless
// of the containing allocation's observed modulo-16 address.
type axpyAlignmentStorage struct {
	a   [128]complex128
	pad uint64
	b   [128]complex128
}

func axpyAlignmentArray(t *testing.T, mod uintptr) []complex128 {
	t.Helper()
	owner := new(axpyAlignmentStorage)
	for _, v := range [][]complex128{owner.a[:], owner.b[:]} {
		address := uintptr(unsafe.Pointer(&v[0]))
		if address%16 == mod {
			if address%uintptr(unsafe.Alignof(complex128(0))) != 0 {
				t.Fatal("invalid Go complex128 alignment")
			}
			return v
		}
	}
	t.Fatalf("ordinary storage did not provide modulo16=%d", mod)
	return nil
}

func axpyAlignmentCall(op string, alpha complex128, x, y, dst []complex128, n int, steps [3]int, starts [3]int) {
	switch op {
	case "Unitary":
		AxpyUnitary(alpha, x[:n], y[:n])
	case "UnitaryTo":
		AxpyUnitaryTo(dst[:n], alpha, x[:n], y[:n])
	case "Inc":
		AxpyInc(alpha, x, y, uintptr(n), uintptr(steps[0]), uintptr(steps[1]), uintptr(starts[0]), uintptr(starts[1]))
	case "IncTo":
		AxpyIncTo(dst, uintptr(steps[2]), uintptr(starts[2]), alpha, x, y, uintptr(n), uintptr(steps[0]), uintptr(steps[1]), uintptr(starts[0]), uintptr(starts[1]))
	default:
		panic("unknown AXPY fixture operation")
	}
}

// The dyadic component oracle has no SIMD, unsafe arithmetic, FMA, or call to
// another AXPY routine. Group snapshots preserve the historical ASM's four
// input reads before stores for partial aliases/repeated write indices. That
// existing behavior need not equal a sequential noasm loop for such aliases.
func axpyAlignmentOracle(alpha complex128, x, y, dst []complex128, n int, steps, starts [3]int) {
	for k := 0; k < n; {
		block := 1
		if n-k >= 4 {
			block = 4
		}
		var values [4]complex128
		for j := 0; j < block; j++ {
			a, b := x[starts[0]+(k+j)*steps[0]], y[starts[1]+(k+j)*steps[1]]
			r := real(alpha)*real(a) - imag(alpha)*imag(a)
			i := imag(alpha)*real(a) + real(alpha)*imag(a)
			values[j] = complex(r+real(b), i+imag(b))
		}
		for j := 0; j < block; j++ {
			dst[starts[2]+(k+j)*steps[2]] = values[j]
		}
		k += block
	}
}

func TestAxpyOrdinaryGoAlignment(t *testing.T) {
	type layout struct {
		mods  [3]uintptr
		alias string
	}
	var layouts []layout
	for _, x := range []uintptr{0, 8} {
		for _, y := range []uintptr{0, 8} {
			for _, d := range []uintptr{0, 8} {
				layouts = append(layouts, layout{[3]uintptr{x, y, d}, "separate"})
			}
		}
	}
	for _, mod := range []uintptr{0, 8} {
		for _, alias := range []string{"xy", "dx", "dy", "all", "dy+1", "dx+1"} {
			layouts = append(layouts, layout{[3]uintptr{mod, mod, mod}, alias})
		}
	}
	for _, op := range []string{"Unitary", "UnitaryTo", "Inc", "IncTo"} {
		stepsList := [][3]int{{1, 1, 1}}
		if op == "Inc" || op == "IncTo" {
			stepsList = [][3]int{{1, 1, 1}, {2, 3, 2}, {-2, -3, -2}, {2, -3, 1}, {0, 3, 2}, {2, 0, 0}, {0, 0, 0}, {-2, 3, 0}}
		}
		for _, n := range []int{0, 1, 2, 3, 4, 5, 7, 8, 9} {
			for _, steps := range stepsList {
				for _, shape := range layouts {
					t.Run(fmt.Sprintf("%s/n=%d/steps=%d,%d,%d/mod=%d,%d,%d/alias=%s", op, n, steps[0], steps[1], steps[2], shape.mods[0], shape.mods[1], shape.mods[2], shape.alias), func(t *testing.T) {
						steps := steps
						var owners, wantOwners [3][]complex128
						for i := range owners {
							owners[i] = axpyAlignmentArray(t, shape.mods[i])
							for j := range owners[i] {
								owners[i][j] = complex(float64((j+3*i)%11-5)/16, float64((2*j+i)%7-3)/16)
							}
							wantOwners[i] = append([]complex128(nil), owners[i]...)
						}
						which, offsets := [3]int{0, 1, 2}, [3]int{3, 2, 4}
						switch shape.alias {
						case "xy":
							which[1], offsets[1] = 0, offsets[0]
						case "dx":
							which[2], offsets[2] = 0, offsets[0]
						case "dy":
							which[2], offsets[2] = 1, offsets[1]
						case "all":
							which, offsets = [3]int{0, 0, 0}, [3]int{3, 3, 3}
						case "dy+1":
							which[2], offsets[2] = 1, offsets[1]+1
						case "dx+1":
							which[2], offsets[2] = 0, offsets[0]+1
						}
						if op == "Unitary" || op == "Inc" {
							which[2], offsets[2] = which[1], offsets[1]
							steps[2] = steps[1]
						}
						var views, wantViews [3][]complex128
						var starts [3]int
						for i := range views {
							span := 1
							if n > 0 {
								step := steps[i]
								if step < 0 {
									step = -step
								}
								span += (n - 1) * step
							}
							views[i] = owners[which[i]][offsets[i] : offsets[i]+span]
							wantViews[i] = wantOwners[which[i]][offsets[i] : offsets[i]+span]
							if steps[i] < 0 {
								starts[i] = span - 1
							}
							if uintptr(unsafe.Pointer(&views[i][0]))%16 != shape.mods[which[i]] {
								t.Fatal("observed alignment changed")
							}
						}
						const alpha = complex(0.5, -0.25)
						axpyAlignmentOracle(alpha, wantViews[0], wantViews[1], wantViews[2], n, steps, starts)
						axpyAlignmentCall(op, alpha, views[0], views[1], views[2], n, steps, starts)
						for i := range owners {
							for j, got := range owners[i] {
								if got != wantOwners[i][j] {
									t.Fatalf("owner=%d index=%d got=%v want=%v", i, j, got, wantOwners[i][j])
								}
							}
						}
						runtime.KeepAlive(owners)
					})
				}
			}
		}
	}
}

func TestAxpyAlignmentEmpty(t *testing.T) {
	max := ^uintptr(0)
	AxpyInc(1, nil, nil, 0, max, max, max, max)
	AxpyIncTo(nil, max, max, 1, nil, nil, 0, max, max, max, max)
	values := []complex128{1 + 2i, 3 - 4i}
	before := append([]complex128(nil), values...)
	AxpyUnitary(1, nil, values)
	AxpyUnitary(1, values, nil)
	AxpyUnitaryTo(nil, 1, values, values)
	AxpyUnitaryTo(values, 1, nil, values)
	AxpyUnitaryTo(values, 1, values, nil)
	for i, got := range values {
		if got != before[i] {
			t.Fatal("empty call wrote values")
		}
	}
}

// Root runs exactly one original-ASM case in a bounded subprocess with core
// dumps disabled. The opt-in is absent from ordinary repaired validation.
func TestAxpyAlignmentOriginalReproducer(t *testing.T) {
	if os.Getenv("GONUM_Q3_REPRODUCE_ORIGINAL") != "yes" {
		t.Skip("root-only original-ASM fault reproduction")
	}
	op := os.Getenv("GONUM_Q3_OPERATION")
	n, err := strconv.Atoi(os.Getenv("GONUM_Q3_N"))
	if err != nil || (n != 1 && n != 4 && n != 5) {
		t.Fatal("reproducer requires n1/4/5")
	}
	x, y, dst := axpyAlignmentArray(t, 0)[:n], axpyAlignmentArray(t, 8)[:n], axpyAlignmentArray(t, 0)[:n]
	for i := range x {
		x[i], y[i] = 1+2i, 3-1i
	}
	fmt.Printf("Q3_ORIGINAL op=%s n=%d x_mod16=%d y_mod16=%d dst_mod16=%d\n", op, n, uintptr(unsafe.Pointer(&x[0]))%16, uintptr(unsafe.Pointer(&y[0]))%16, uintptr(unsafe.Pointer(&dst[0]))%16)
	axpyAlignmentCall(op, 0.5-0.25i, x, y, dst, n, [3]int{1, 1, 1}, [3]int{})
	runtime.KeepAlive(x)
	runtime.KeepAlive(y)
	runtime.KeepAlive(dst)
}
