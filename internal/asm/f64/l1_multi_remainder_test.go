// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64

import (
	"fmt"
	"math"
	"testing"
)

// The existing integer oracle and original-entry oracle remain authoritative.
// Every possible new remainder position is marked for each native width and
// chain count; 32 is the largest four-chain block, not a runtime width guess.
func TestL1MultiRemainderPositions(t *testing.T) {
	values := []struct {
		name   string
		x      float64
		prefix float64
		exact  bool
		retry  bool
	}{
		{"normal", -2, 0, true, false},
		{"minsub", -math.SmallestNonzeroFloat64, 0, true, false},
		{"minnormal", -math.Float64frombits(1 << 52), 0, true, false},
		// A nonzero leading contribution detects recovery on a sliced suffix.
		{"half-above-with-prefix", math.Nextafter(math.MaxFloat64/2, math.Inf(1)), math.MaxFloat64 / 4, false, true},
		{"max-with-prefix", math.MaxFloat64, math.MaxFloat64 / 4, false, true},
		{"positive-inf", math.Inf(1), 0, false, true},
		{"negative-inf", math.Inf(-1), 0, false, true},
		{"nan", math.Float64frombits(0x7ff8000000000042), 0, false, true},
	}
	for _, base := range []int{128, 256} {
		for r := 0; r < 32; r++ {
			n := base + r
			for _, offset := range []int{0, 1} {
				t.Run(fmt.Sprintf("n%d/offset%d", n, offset), func(t *testing.T) {
					checked, lastIndex, lastValue := 0, 0, ""
					t.Cleanup(func() {
						if t.Failed() {
							t.Logf("last remainder case: n=%d offset=%d index=%d value=%s", n, offset, lastIndex, lastValue)
						}
					})
					// The r=0 control executes too, with a marker in the main block.
					first := base
					if r == 0 {
						first = n - 1
					}
					for i := first; i < n; i++ {
						for _, value := range values {
							backing := make([]float64, n+offset+1)
							if offset != 0 {
								backing[0] = math.Float64frombits(0x7ff8000000000051)
							}
							backing[len(backing)-1] = math.Float64frombits(0x7ff8000000000052)
							x := backing[offset : offset+n : offset+n]
							x[0], x[i] = value.prefix, value.x
							saved := make([]uint64, len(backing))
							for j, v := range backing {
								saved[j] = math.Float64bits(v)
							}
							lastIndex, lastValue = i, value.name
							l1MultiCheck(t, x, L1NormSIMD(x), value.exact, value.retry)
							l1MultiCheck(t, x, L1NormIncSIMD(x, n, 1), value.exact, value.retry)
							checked++
							for j, v := range backing {
								if math.Float64bits(v) != saved[j] {
									t.Fatalf("input/canary changed at %d", j)
								}
							}
						}
					}
					t.Logf("L1_REMAINDER_CHECKED n=%d offset=%d cases=%d", n, offset, checked)
				})
			}
		}
	}
}
