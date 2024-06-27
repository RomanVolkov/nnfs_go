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
