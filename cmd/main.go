package main

import "main/models"

func main() {
	// models.RunRegressionModel()
	// models.RunBinaryModel()
	models.RunCategorialModel()
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
}
