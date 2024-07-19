package utils

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func PrintDims(r int, c int) {
	fmt.Println("Dims", r, c)
}

func PrintDense(value *mat.Dense) {
	fmt.Println("================")
	fmt.Println(value.Dims())
	r, c := value.Dims()
	goString := "value := float64{"
	pythonString := "return np.array(["

	for _, v := range value.RawMatrix().Data {
		goString += fmt.Sprintf("%f", v) + ", "
		pythonString += fmt.Sprintf("%f", v) + ", "
	}

	goString += "}\nresult := mat.NewDesne(" + fmt.Sprint(r) + ", " + fmt.Sprint(c) + ", nil)"
	pythonString += "]).repshape(" + fmt.Sprint(r) + ", " + fmt.Sprint(c) + ")"
	fmt.Println(goString)
	fmt.Println(pythonString)
}

func CompareDims(lhs *mat.Dense, rhs *mat.Dense) bool {
	l_r, l_c := lhs.Dims()
	r_r, r_c := rhs.Dims()

	if l_r != r_r {
		return false
	}
	return l_c == r_c
}

func MakeRange(size int) []int {
	indexes := make([]int, size)
	for i := 0; i < size; i++ {
		indexes[i] = i
	}
	return indexes
}
