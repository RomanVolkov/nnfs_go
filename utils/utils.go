package utils

import (
	"errors"
	"fmt"
	"golang.org/x/image/draw"
	"image"
	"image/color"

	"gonum.org/v1/gonum/mat"
)

func PrintDims(r int, c int) {
	fmt.Println("Dims:", r, c)
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

func CompareDims(lhs *mat.Matrix, rhs *mat.Matrix) bool {
	l_r, l_c := (*lhs).Dims()
	r_r, r_c := (*rhs).Dims()

	return l_c == r_c && l_r == r_r
}

func MakeRange(size int) []int {
	indexes := make([]int, size)
	for i := 0; i < size; i++ {
		indexes[i] = i
	}
	return indexes
}

func ConvertIntoGrayscale(src image.Image, width int, height int) image.Image {
	dst := image.NewGray(image.Rect(0, 0, width, height))
	draw.NearestNeighbor.Scale(dst, dst.Rect, src, src.Bounds(), draw.Over, nil)
	return dst
}

func NormalizeGrascaleImageData(img image.Image, invertColor bool) ([]float64, error) {
	rows := img.Bounds().Max.X
	cols := img.Bounds().Max.Y

	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			gray, ok := img.At(i, j).(color.Gray)
			if ok {
				y := gray.Y
				if invertColor {
					y = 255 - y
				}
				// convert the range into -1.0...1.0
				data[i*rows+j] = (float64(y) - 127.5) / 127.5
			} else {
				return nil, errors.New("cannot take Grayscale color")
			}
		}
	}

	return data, nil
}
