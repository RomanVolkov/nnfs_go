package main

import (
	"fmt"
	"log"
	"main/dataset"
	// "gonum.org/v1/gonum/mat"
)

func main() {
	// models.RunRegressionModel()
	// models.RunBinaryModel()
	// models.RunCategorialModel()
	// err := utils.DownloadFile("https://nnfs.io/datasets/fashion_mnist_images.zip", "./assets/fashion_mnist_images.zip")
	// if err != nil {
	// 	log.Fatal(err)
	// 	return
	// }
	//
	// err = utils.UnzipDataset("./assets/fashion_mnist_images.zip", "./assets/fashion_mnist_images/")
	// if err != nil {
	// 	log.Fatal(err)
	// 	return
	// }
	// path := "./assets/fashion_mnist_images/train/0/0001.png"
	// file, err := os.Open(path)
	// if err != nil {
	// 	panic(err)
	// }
	// defer file.Close()
	//
	// img, err := png.Decode(file)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(img.Bounds())
	// fmt.Println(img.At(10, 10))

	d := dataset.FashionMNISTDataset{}
	_, target, error := d.TrainingDataset()
	if error != nil {
		log.Fatal(error)
	}

	fmt.Println(target.At(10, 0))

}
