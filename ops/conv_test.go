package ops_test

import (
	"main/ops"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCorrelate2DOps(t *testing.T) {
	input := mat.NewDense(2, 2, []float64{0, 1, 3, 4})
	kernel := mat.NewDense(2, 2, []float64{0, 1, 2, 3})

	result, _ := ops.Correlate2dOps(input, kernel)
	if result != 19 {
		t.Fatalf("Correlate2d failed with result: %v", result)
	}
}

func TestCorrelate2DOpsSizeMismatch(t *testing.T) {
	input := mat.NewDense(2, 2, []float64{0, 1, 3, 4})
	kernel := mat.NewDense(1, 3, []float64{0, 1, 2})

	_, err := ops.Correlate2dOps(input, kernel)
	if err == nil {
		t.Fatal("Missing error")
	}
}

func TestCorrelate2DValid(t *testing.T) {
	input := *mat.NewDense(3, 3, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8})
	kernel := *mat.NewDense(2, 2, []float64{0, 1, 2, 3})

	result, _ := ops.Correlate2dValid(input, kernel)

	if !compare(result.RawMatrix().Data, []float64{19, 25, 37, 43}) {
		t.Fatal("Nope")
	}
}

func compare(l []float64, r []float64) bool {
	if len(l) != len(r) {
		return false
	}

	for i := 0; i < len(l); i++ {
		if l[i] != r[i] {
			return false
		}
	}

	return true
}
