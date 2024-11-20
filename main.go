package main

import (
	"fmt"
	"image/png"
	"log"
	"main/dataset"
	"main/model"
	"main/models"
	"main/utils"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func testModelV1() {
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

	// input, _ := os.Open("./assets/tshirt.png")
	input, _ := os.Open("./assets/pants.png")
	defer input.Close()

	src, _ := png.Decode(input)
	dst := utils.ConvertIntoGrayscale(src, 28, 28)

	data, err := utils.NormalizeGrascaleImageData(dst, true)
	if err != nil {
		log.Fatal(err)
	}

	classes := map[int]string{
		0: "T-shirt/top",
		1: "Trouser",
		2: "Pullover",
		3: "Dress",
		4: "Coat",
		5: "Sandal",
		6: "Shirt",
		7: "Sneaker",
		8: "Bag",
		9: "Ankle boot",
	}

	inputData := mat.NewDense(1, 28*28, data)
	predictions = loadedModel.Predict(inputData, nil)
	fmt.Println("================================")
	fmt.Println("calling prediction")
	fmt.Println(mat.Formatted(&predictions))
	classIndex := floats.MaxIdx(predictions.RawMatrix().Data)
	fmt.Println(classes[classIndex])

}

func main() {
	// models.TrainModels()
	models.LoadModels()
}
