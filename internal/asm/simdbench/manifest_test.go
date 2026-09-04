// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package simdbench

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
)

func TestAMD64SIMDCandidateCoverage(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("could not locate manifest test")
	}
	asmRoot := filepath.Dir(filepath.Dir(filename))

	want := make([]string, len(AMD64Assembly))
	for i, entry := range AMD64Assembly {
		want[i] = entry.Package + "." + entry.Symbol + "SIMD"
	}
	sort.Strings(want)

	var got []string
	directSIMD := make(map[string]bool)
	calls := make(map[string]map[string]bool)
	fset := token.NewFileSet()
	for _, pkg := range []string{"c128", "c64", "f32", "f64"} {
		path := filepath.Join(asmRoot, pkg, "simd.go")
		checkPortableSIMDBuildConstraint(t, path)
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Body == nil {
				continue
			}
			name := pkg + "." + function.Name.Name
			if ast.IsExported(function.Name.Name) && strings.HasSuffix(function.Name.Name, "SIMD") {
				got = append(got, name)
			}
			calls[name] = make(map[string]bool)
			ast.Inspect(function.Body, func(node ast.Node) bool {
				call, ok := node.(*ast.CallExpr)
				if !ok {
					return true
				}
				switch target := call.Fun.(type) {
				case *ast.Ident:
					calls[name][pkg+"."+target.Name] = true
				case *ast.SelectorExpr:
					if receiver, ok := target.X.(*ast.Ident); ok && receiver.Name == "simd" {
						directSIMD[name] = true
					}
				}
				return true
			})
		}
	}
	sort.Strings(got)
	if diff := diffStrings(want, got); diff != "" {
		t.Fatalf("AMD64 SIMD candidate coverage mismatch:\n%s", diff)
	}
	for _, candidate := range want {
		if !reachesSIMD(candidate, directSIMD, calls, make(map[string]bool)) {
			t.Errorf("%s does not reach a simd package operation", candidate)
		}
	}
}

func checkPortableSIMDBuildConstraint(t *testing.T, path string) {
	t.Helper()
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var expression constraint.Expr
	for line := range strings.SplitSeq(string(content), "\n") {
		if !strings.HasPrefix(line, "//go:build ") {
			continue
		}
		expression, err = constraint.Parse(line)
		if err != nil {
			t.Fatalf("parse build constraint in %s: %v", path, err)
		}
		break
	}
	if expression == nil {
		t.Fatalf("%s has no build constraint", path)
	}
	for _, arch := range []string{"amd64", "arm64", "riscv64"} {
		if !expression.Eval(func(tag string) bool {
			return tag == "go1.27" || tag == "goexperiment.simd" || tag == arch
		}) {
			t.Errorf("%s is not portable to %s", path, arch)
		}
	}
	for _, optOut := range []string{"safe", "noasm", "gccgo"} {
		if expression.Eval(func(tag string) bool {
			return tag == "go1.27" || tag == "goexperiment.simd" || tag == "amd64" || tag == optOut
		}) {
			t.Errorf("%s does not honor %s", path, optOut)
		}
	}
}

func TestAMD64BenchmarkCoverage(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("could not locate manifest test")
	}
	path := filepath.Join(filepath.Dir(filename), "coverage_simd_test.go")
	file, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	var got []string
	ast.Inspect(file, func(node ast.Node) bool {
		clause, ok := node.(*ast.CaseClause)
		if !ok {
			return true
		}
		for _, expression := range clause.List {
			literal, ok := expression.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				continue
			}
			value, err := strconv.Unquote(literal.Value)
			if err == nil && strings.Contains(value, ".") {
				got = append(got, value)
			}
		}
		return true
	})
	want := make([]string, len(AMD64Assembly))
	for i, entry := range AMD64Assembly {
		want[i] = entry.Package + "." + entry.Symbol
	}
	sort.Strings(want)
	sort.Strings(got)
	if diff := diffStrings(want, got); diff != "" {
		t.Fatalf("AMD64 benchmark coverage mismatch:\n%s", diff)
	}
}

func reachesSIMD(name string, direct map[string]bool, calls map[string]map[string]bool, visiting map[string]bool) bool {
	if direct[name] {
		return true
	}
	if visiting[name] {
		return false
	}
	visiting[name] = true
	defer delete(visiting, name)
	for callee := range calls[name] {
		if reachesSIMD(callee, direct, calls, visiting) {
			return true
		}
	}
	return false
}

func TestAMD64AssemblyManifest(t *testing.T) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("could not locate manifest test")
	}
	asmRoot := filepath.Dir(filepath.Dir(filename))

	want := make([]string, len(AMD64Assembly))
	for i, entry := range AMD64Assembly {
		want[i] = entry.Package + "." + entry.Symbol
	}
	sort.Strings(want)
	for i := 1; i < len(want); i++ {
		if want[i] == want[i-1] {
			t.Fatalf("duplicate manifest entry %q", want[i])
		}
	}

	var got []string
	for _, pkg := range []string{"c128", "c64", "f32", "f64"} {
		files, err := filepath.Glob(filepath.Join(asmRoot, pkg, "*_amd64.s"))
		if err != nil {
			t.Fatal(err)
		}
		for _, path := range files {
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			scanner := bufio.NewScanner(f)
			for scanner.Scan() {
				line := scanner.Text()
				if !strings.HasPrefix(line, "TEXT ·") {
					continue
				}
				name := strings.TrimPrefix(line, "TEXT ·")
				if end := strings.IndexByte(name, '('); end >= 0 {
					name = name[:end]
				}
				got = append(got, pkg+"."+name)
			}
			if err := scanner.Err(); err != nil {
				f.Close()
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
		}
	}
	sort.Strings(got)
	if diff := diffStrings(want, got); diff != "" {
		t.Fatalf("AMD64 assembly manifest mismatch:\n%s", diff)
	}
}

func diffStrings(want, got []string) string {
	var missing, extra []string
	wantSet := make(map[string]bool, len(want))
	gotSet := make(map[string]bool, len(got))
	for _, value := range want {
		wantSet[value] = true
	}
	for _, value := range got {
		gotSet[value] = true
	}
	for _, value := range want {
		if !gotSet[value] {
			missing = append(missing, value)
		}
	}
	for _, value := range got {
		if !wantSet[value] {
			extra = append(extra, value)
		}
	}
	if len(missing) == 0 && len(extra) == 0 {
		return ""
	}
	return fmt.Sprintf("missing=%v extra=%v", missing, extra)
}
