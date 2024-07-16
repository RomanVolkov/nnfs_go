package dataset

import (
	"errors"
	"image/color"
	"image/png"
	"io/fs"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

const (
	rootPath     = "./assets/fashion_mnist_images/"
	trainingPath = rootPath + "train/"
	testingPath  = rootPath + "test/"
)

type FashionMNISTDataset struct {
}

func (f *FashionMNISTDataset) TrainingDataset() (*mat.Dense, *mat.Dense, error) {
	return f.loadDataset(trainingPath)
}
func (f *FashionMNISTDataset) TestingDataset() (*mat.Dense, *mat.Dense, error) {
	return f.loadDataset(testingPath)
}

func (f *FashionMNISTDataset) loadDataset(datasetPath string) (*mat.Dense, *mat.Dense, error) {
	imagesData := make([][]float64, 0)
	labels := make([]float64, 0)

	err := filepath.WalkDir(datasetPath, func(rootPath string, d fs.DirEntry, err error) error {
		if d.IsDir() {
			return nil
		}
		labelStr := path.Base(filepath.Dir(rootPath))
		label, err := strconv.Atoi(labelStr)
		if err != nil {
			return err
		}
		imageData, err := f.loadImage(rootPath)
		if err != nil {
			return err
		}

		imagesData = append(imagesData, imageData)
		labels = append(labels, float64(label))

		return nil
	})

	if err != nil {
		return nil, nil, err
	}

	imagesCount := len(imagesData)
	if imagesCount != len(labels) {
		return nil, nil, errors.New("inconsistency between images number and labels")
	}

	if imagesCount == 0 {
		return nil, nil, errors.New("Empty dataset")
	}

	dataLen := len(imagesData[0])
	resultData := mat.NewDense(imagesCount, dataLen, nil)
	resultLabels := mat.NewDense(imagesCount, 1, nil)
	shuffledIndexes := shuffled(makeRange(imagesCount))

	for i := 0; i < imagesCount; i++ {
		idx := shuffledIndexes[i]
		resultLabels.Set(idx, 0, labels[i])
		resultData.SetRow(idx, imagesData[i])
	}

	return resultData, resultLabels, nil
}

func (f *FashionMNISTDataset) loadImage(path string) ([]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, err := png.Decode(file)
	if err != nil {
		return nil, err
	}

	rows := img.Bounds().Max.X
	cols := img.Bounds().Max.Y

	data := make([]float64, rows*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			gray, ok := img.At(i, j).(color.Gray)
			if ok {
				y := gray.Y
				data[i*rows+j] = (float64(y) - 127.5) / 127.5
			} else {
				return nil, errors.New("cannot take Grayscale color")
			}

		}
	}

	return data, nil
}

func makeRange(lenght int) []int {
	values := make([]int, lenght)
	for i := 0; i < lenght; i++ {
		values[i] = i
	}
	return values
}

func shuffled(values []int) []int {
	shuffled := make([]int, len(values))
	copy(shuffled, values)
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	return shuffled
}
