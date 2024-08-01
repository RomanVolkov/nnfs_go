package main

import (
	"fmt"
	"image/png"
	"log"
	"main/dataset"
	"main/model"
	"main/utils"
	"os"

	"gonum.org/v1/gonum/mat"
)

func main() {
	dataProvider := model.JSONModelDataProvider{}
	ds := dataset.FashionMNISTDataset{}

	loadedModel, err := dataProvider.Load("./assets/fashion.json")
	if err != nil {
		log.Panic(err)
	}
	x_val, y_val, err := ds.TestingDataset()
	if err != nil {
		log.Fatal(err)
	}
	validationData := model.ModelData{X: *x_val, Y: *y_val}
	batchSize := 128

	loadedModel.Evaluate(validationData, &batchSize)

	_, c := x_val.Dims()
	testX := mat.NewDense(1, c, x_val.RawRowView(0))

	predictions := loadedModel.Predict(testX, nil)
	fmt.Println(mat.Formatted(&predictions))

	input, _ := os.Open("./assets/tshirt.png")
	defer input.Close()

	target, _ := os.Create("./assets/tshirt-g.png")
	defer target.Close()

	src, _ := png.Decode(input)
	dst := utils.ConvertIntoGrayscale(src, 28, 28)

	png.Encode(target, dst)

}
