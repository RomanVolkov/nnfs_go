package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"log"
	"main/model"
	"main/utils"
	"net/http"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var loadedModel model.Model

func main() {
	err := loadModel()
	if err != nil {
		log.Fatal(err)
	}
	// register for endpoints

	http.HandleFunc("/predict", predictionHandler)
	log.Println("Running")
	http.ListenAndServe(":8090", nil)
}

// MARK: - Endpoints

func predictionHandler(w http.ResponseWriter, req *http.Request) {
	if req.Method != "POST" {
		http.Error(w, "incorrect methond", http.StatusInternalServerError)
		return
	}

	file, _, err := req.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	img, err := png.Decode(file)
	defer file.Close()

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	class, className, err := prediction(img)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	respt := struct {
		Class int    `json:"classIndex"`
		Name  string `json:"className"`
	}{
		Class: class,
		Name:  className,
	}

	d, err := json.Marshal(respt)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Fprintln(w, string(d))
}

// MARK: - Processing

func loadModel() error {
	dataProvider := model.JSONModelDataProvider{}
	m, err := dataProvider.Load("./assets/fashion.json")
	if err != nil {
		return err
	}
	loadedModel = *m
	return nil
}

func prediction(img image.Image) (int, string, error) {
	dst := utils.ConvertIntoGrayscale(img, 28, 28)

	data, err := utils.NormalizeGrascaleImageData(dst, true)
	if err != nil {
		return -1, "", err
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
	predictions := loadedModel.Predict(inputData, nil)
	classIndex := floats.MaxIdx(predictions.RawMatrix().Data)

	return classIndex, classes[classIndex], nil
}
