// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "testing"

func TestDtgsenWorkspaceQueryRequiresOutputs(t *testing.T) {
	for _, test := range []struct {
		name  string
		work  []float64
		iwork []int
		want  string
	}{
		{name: "Float", iwork: make([]int, 1), want: shortWork},
		{name: "Integer", work: make([]float64, 1), want: shortIWork},
	} {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if got := recover(); got != test.want {
					t.Fatalf("panic=%v, want %q", got, test.want)
				}
			}()
			Implementation{}.Dtgsen(0, false, false, nil, 0,
				nil, 1, nil, 1, nil, nil, nil, nil, 1, nil, 1,
				test.work, -1, test.iwork, -1)
		})
	}
}
