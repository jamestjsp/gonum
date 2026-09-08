// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"math"
	"simd"
	"simd/archsimd"
	"testing"
)

func normPeelAvailable() bool {
	return !simd.Emulated() && simd.VectorBitSize() >= 256 && archsimd.X86.AVX2() && archsimd.X86.FMA()
}

func TestNormPeelSeedLaneBits(t *testing.T) {
	w := 3 * 0x1p-539
	p := float64(w * w)
	r := math.FMA(w, w, -p)
	if math.Float64bits(p) != 1 || math.Float64bits(r) != 1<<63 {
		t.Fatalf("independent signed-zero witness: product=%x residual=%x", math.Float64bits(p), math.Float64bits(r))
	}
	if !normPeelAvailable() {
		t.Skip("native first-block path unavailable; scalar witness passed")
	}
	values := []float64{w, -w, 0, math.Copysign(0, -1), math.SmallestNonzeroFloat64,
		-math.SmallestNonzeroFloat64, 0x1p-537, 0x1p-500, 0x1p-600, 1, -1, 1 + 0x1p-27, 0x1p450}
	positive, negative := false, false
	for i := uint64(0); i < 128; i++ {
		v := math.Float64frombits(0x3ff0000000000000 | (i*0x9e3779b97f4a7c15)&0xfffffffffffff)
		values = append(values, v, -v)
		e := math.FMA(v, v, -float64(v*v))
		positive = positive || e > 0
		negative = negative || e < 0
	}
	if !positive || !negative {
		t.Fatal("deterministic cases must contain both nonzero residual signs")
	}
	for i := 0; i < len(values); i += 4 {
		var lanes [4]float64
		for j := range lanes {
			lanes[j] = values[(i+j)%len(values)]
		}
		checkNormPeelLaneBits(t, lanes)
	}
}

// Keep vector work in an out-of-line checked callee, after the driver's feature
// admission. This also checks the actual seed helper, rather than a copy.
//
//go:noinline
func checkNormPeelLaneBits(t *testing.T, lanes [4]float64) {
	v := archsimd.LoadFloat64x4Array(&lanes)
	sign := archsimd.BroadcastUint64x4(1 << 63)
	var zero archsimd.Float64x4
	p, c := normSeedNative256(v, sign)
	op, oc := normSquareNative256(v, zero, zero, sign)
	var products, corrections, oldProducts, oldCorrections [4]float64
	p.StoreArray(&products)
	c.StoreArray(&corrections)
	op.StoreArray(&oldProducts)
	oc.StoreArray(&oldCorrections)
	archsimd.ClearAVXUpperBits()
	for i, x := range lanes {
		wantP := float64(x * x)
		wantC := float64(0) + math.FMA(x, x, -wantP)
		if math.Float64bits(products[i]) != math.Float64bits(wantP) ||
			math.Float64bits(corrections[i]) != math.Float64bits(wantC) ||
			math.Float64bits(products[i]) != math.Float64bits(oldProducts[i]) ||
			math.Float64bits(corrections[i]) != math.Float64bits(oldCorrections[i]) {
			t.Fatalf("x=%x seed=(%x,%x) old=(%x,%x) scalar=(%x,%x)", math.Float64bits(x),
				math.Float64bits(products[i]), math.Float64bits(corrections[i]),
				math.Float64bits(oldProducts[i]), math.Float64bits(oldCorrections[i]),
				math.Float64bits(wantP), math.Float64bits(wantC))
		}
	}
}

func TestNormPeelPublicExactParity(t *testing.T) {
	if !normPeelAvailable() {
		t.Skip("native first-block path unavailable")
	}
	sizes := []int{47, 48, 49, 63, 64, 65, 129, 4097}
	for n := 16; n <= 33; n++ {
		sizes = append(sizes, n)
	}
	for _, n := range sizes {
		for _, inc := range []int{2, 3, 7, 16} {
			for distribution := 0; distribution < 4; distribution++ {
				x := make([]float64, (n-1)*inc+1)
				for i := range x {
					x[i] = math.NaN()
				}
				for i := 0; i < n; i++ {
					v := 1 + float64(i%31)/64
					switch distribution {
					case 1:
						if i != 0 {
							v = 0x1p-27
						}
					case 2:
						v = 1 + float64(i%63)*0x1p-27
					case 3:
						if i%3 != 0 {
							v = 3 * 0x1p-539
						}
					}
					x[i*inc] = v
				}
				got := L2NormIncSIMD(x, uintptr(n), uintptr(inc))
				old := l2NormIncUnpeeledReference(x, uintptr(n), uintptr(inc))
				if math.Float64bits(got) != math.Float64bits(old) {
					t.Fatalf("n=%d inc=%d distribution=%d got=%x old=%x", n, inc, distribution, math.Float64bits(got), math.Float64bits(old))
				}
				checkNativeNormULP(t, got, nativeNormReference(x, n, inc))
			}
		}
	}
}

func TestNormPeelExceptionalPositions(t *testing.T) {
	if !normPeelAvailable() {
		t.Skip("native first-block path unavailable")
	}
	specials := []float64{0, math.Copysign(0, -1), math.SmallestNonzeroFloat64,
		3 * 0x1p-539, -3 * 0x1p-539, 0x1p-600, 0x1p1000, math.Inf(1), math.Inf(-1), math.NaN()}
	for _, n := range []int{16, 17, 31, 32, 33, 65} {
		for _, inc := range []int{2, 3, 7, 16} {
			for _, special := range specials {
				for position := 0; position < n; position++ {
					x := make([]float64, (n-1)*inc+1)
					for i := range x {
						x[i] = math.NaN()
					}
					for i := 0; i < n; i++ {
						x[i*inc] = 1
					}
					x[position*inc] = special
					got := L2NormIncSIMD(x, uintptr(n), uintptr(inc))
					old := l2NormIncUnpeeledReference(x, uintptr(n), uintptr(inc))
					if !(math.IsNaN(got) && math.IsNaN(old)) && math.Float64bits(got) != math.Float64bits(old) {
						t.Fatalf("n=%d inc=%d position=%d special=%g got=%x old=%x", n, inc, position, special, math.Float64bits(got), math.Float64bits(old))
					}
					if math.IsNaN(special) || math.IsInf(special, 0) {
						checkSIMDNorm(t, got, math.Abs(special))
					} else {
						checkSIMDNorm(t, got, nativeNormReference(x, n, inc))
					}
					if math.IsInf(special, 0) {
						x[((position+1)%n)*inc] = math.NaN()
						if !math.IsNaN(L2NormIncSIMD(x, uintptr(n), uintptr(inc))) {
							t.Fatal("mixed Inf/NaN must retain NaN precedence")
						}
					}
				}
			}
		}
	}
	for _, v := range specials[:6] {
		x := make([]float64, 31*3+1)
		for i := 0; i < 32; i++ {
			x[i*3] = v
		}
		got := L2NormIncSIMD(x, 32, 3)
		old := l2NormIncUnpeeledReference(x, 32, 3)
		if math.Float64bits(got) != math.Float64bits(old) {
			t.Fatalf("homogeneous tiny/signed-zero input %g: %x != %x", v, math.Float64bits(got), math.Float64bits(old))
		}
		checkSIMDNorm(t, got, nativeNormReference(x, 32, 3))
	}
}
