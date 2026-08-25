// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import "testing"

func TestDtgex2WorkspaceQueryRequiresOutput(t *testing.T) {
	defer func() {
		if got := recover(); got != shortWork {
			t.Fatalf("panic=%v, want %q", got, shortWork)
		}
	}()
	Implementation{}.Dtgex2(false, false, 2,
		nil, 2, nil, 2, nil, 1, nil, 1, 0, 1, 1, nil, -1)
}
