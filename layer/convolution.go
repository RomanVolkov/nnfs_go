package layer

import (
	"log"
	"main/ops"
	"math/rand"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type ConvolutionLayer struct {
	// shape of the input data. E.g., Grayscale image is (1, x, x); RGB is (3, x, x)
	InputShape InputShape
	// 2D array - rows -> number of samples; cols -> input sample data of InputShape.Depths * InputShape.Width * Inputs.Height
	Inputs mat.Dense

	// Depths of Convolution Layer. Defines number of kernels
	Depths int

	// Defines the side of kenlel matrix. Kernels are always square, so any kenle will be (KernelSize, KernelSize)
	KernelSize int

	// Defines the shape of Convolution layer
	// In this case Depths is equal to Depths of Convolution
	// W&H is calculated based on input shape and kernel size
	OutputShape InputShape

	// Defines shape of DS to store kernels data corresponding to Convolution Depths and Input Depths
	KernelShape KernelShape

	// Trainable params
	// holds kernel data according to KernelShape
	Kernels [][]mat.Dense
	// holds biases values according to OutputShape
	Biases []mat.Dense

	// Outputs

	// defines 2D matrix of Convolution results
	// rows -> number of input samples
	// cols -> defines one convolution data. Should be OutputShape.Depths * OutputShape.Width * OutputShape.Height
	Output mat.Dense

	// defines 2D matrix of derivatives with respect to input to pass into next Backward methond
	// rows -> number of input samples
	// cols -> should be of InputShape shape
	DInputs mat.Dense

	// derivatives to adjust Kernels
	dKernels [][][]mat.Dense
	// derivatives to adjust Biases
	dvalues mat.Dense
}

type InputShape struct {
	Depths int `json:"depths"`
	Height int `json:"height"`
	Width  int `json:"width"`
}

func (shape InputShape) TotalSize() int {
	return shape.Depths * shape.Height * shape.Width
}

type KernelShape struct {
	// defiens the total number of kernels
	Depths int `json:"depths"`
	// defines depths of kernel itself. E.g., for Grayscale input InputDepths == 1 while for RGB InputDepths == 3
	InputDepths int `json:"input_depths"`
	// kernelSize
	Height int `json:"height"`
	// kernelSize
	Width int `json:"width"`
}

func (shape KernelShape) TotalSize() int {
	return shape.Depths * shape.InputDepths * shape.Height * shape.Width
}

func (layer *ConvolutionLayer) Initialization(inputShape InputShape, convolutionDepths int, kernelSize int) *ConvolutionLayer {
	// I will need
	// 1. something that described input shape
	// 2. depths of convolution == number of kernels
	// 3. and kernel size
	layer.InputShape = inputShape
	layer.Depths = convolutionDepths
	layer.KernelSize = kernelSize

	// output is defined by convolution depths and relation between input size and kernel size
	// this is simplified version of computation that uses "1" as stride and "0" padding
	layer.OutputShape = InputShape{
		Depths: convolutionDepths,
		Height: inputShape.Height - kernelSize + 1,
		Width:  inputShape.Width - kernelSize + 1,
	}

	layer.KernelShape = KernelShape{
		Depths:      convolutionDepths,
		InputDepths: inputShape.Depths,
		Height:      kernelSize,
		Width:       kernelSize,
	}

	// Kernels

	layer.Kernels = make([][]mat.Dense, layer.Depths)
	for i := 0; i < layer.KernelShape.Depths; i++ {
		layer.Kernels[i] = make([]mat.Dense, layer.InputShape.Depths)
		for j := 0; j < layer.KernelShape.InputDepths; j++ {
			layer.Kernels[i][j] = *mat.NewDense(layer.KernelShape.Height, layer.KernelShape.Width, nil)

			r, c := layer.Kernels[i][j].Dims()
			for ii := 0; ii < r; ii++ {
				for jj := 0; jj < c; jj++ {
					layer.Kernels[i][j].Set(ii, jj, rand.NormFloat64())
				}
			}
		}
	}

	// Biases
	layer.Biases = make([]mat.Dense, layer.OutputShape.Depths)
	for i := 0; i < layer.OutputShape.Depths; i++ {
		layer.Biases[i] = *mat.NewDense(layer.OutputShape.Height, layer.OutputShape.Width, nil)

		r, c := layer.Biases[i].Dims()
		for ii := 0; ii < r; ii++ {
			for jj := 0; jj < c; jj++ {
				layer.Biases[i].Set(ii, jj, rand.NormFloat64())
			}
		}
	}

	return layer
}

func (layer *ConvolutionLayer) LoadFromParams(inputShape InputShape, depths int, kernelSize int, outputShape InputShape, kernelShape KernelShape, kernels [][]mat.Dense, biases []mat.Dense) {
	layer.InputShape = inputShape
	layer.Depths = depths
	layer.KernelSize = kernelSize
	layer.OutputShape = outputShape
	layer.KernelShape = kernelShape
	layer.Kernels = kernels
	layer.Biases = biases
}

func (layer *ConvolutionLayer) Forward(inputs *mat.Dense, isTraining bool) {
	layer.Inputs = *inputs
	inputSampleCount, _ := inputs.Dims()

	// validate input shape
	row := inputs.RawRowView(0)
	if len(row) != layer.InputShape.TotalSize() {
		log.Fatalf("Unexpected input size: %v with InputShape: %v", len(row), layer.InputShape)
	}

	// now we need to create Output
	// Rows -> number of input samples. we can get from inputs.Dims() -> R
	// Cols -> it is defined by layer.OutputShape.TotalSize()

	layer.Output = *mat.NewDense(inputSampleCount, layer.OutputShape.TotalSize(), nil)
	layer.Output.Zero()

	allBiases := layer.AllBiases()
	if len(allBiases) != layer.OutputShape.TotalSize() {
		log.Fatalf("Unexpected length of biases: %v with OutputShape: %v", len(allBiases), layer.OutputShape)
	}
	for i := 0; i < inputSampleCount; i++ {
		// copy all biases values into Output
		// then we will start adding results of convolution into output that already have biases
		layer.Output.SetRow(i, allBiases)
	}

	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	// going thought all input samples
	for k := 0; k < inputSampleCount; k++ {
		wg.Add(1)
		go func(layer *ConvolutionLayer, m *sync.Mutex, k int) {
			defer wg.Done()

			// for convenience raw data from one sample from inputs is converted according to InputShape
			inputSample := ConvertSampleData(inputs.RawRowView(k), layer.InputShape)

			// going thought all kernels within convolution layer
			for i := 0; i < layer.KernelShape.Depths; i++ {
				// going via all sub-kernels (values from 1 kernel for each input channel)
				for j := 0; j < layer.KernelShape.InputDepths; j++ {
					// convolution result for every sub-kernel
					convResult, err := ops.Correlate2dValid(inputSample[j], layer.Kernels[i][j])
					if err != nil {
						log.Fatal(err)
					}

					// Output has OutputShape. Meaning we have number of outputs for specific sample equal to depths of convolution
					// so the layer.Output row will be OutputShape.TotalSize(): depths * output size
					// convResult will be used for the same output InputDepths number of times.
					// e.g., we will write into the same self.output[i] (python) that many times as input data has input channels

					// so here I need to sum convResult into layer.Output.Row(i)[slice for i-th output]
					// in total this operation will happen ConvolutionLayer::depths * InputShape::Depths ->
					// depths of convolution layer * depths of input data (e.g, 1 for Grayscale; 3 for RGB)

					// since Output is stored as 1D array it will be easier to do sum
					rawConvResult := convResult.RawMatrix().Data
					m.Lock()
					for rawI := 0; rawI < len(rawConvResult); rawI++ {
						outputIdx := i*layer.OutputShape.Height*layer.OutputShape.Width + rawI
						prevValue := layer.Output.At(k, outputIdx)
						layer.Output.Set(k, outputIdx, prevValue+rawConvResult[rawI])
					}
					m.Unlock()
				}
			}
		}(layer, &m, k)
	}

	wg.Wait()

	// Output shape will be number of input samples * layer.OutputShape.TotalSize()
}

// dvalues comes form max pooling (well, via relu)
func (layer *ConvolutionLayer) Backward(dvalues *mat.Dense) {
	layer.dvalues = *mat.DenseCopyOf(dvalues)
	// okay, now I can start building Backward pass
	// 1. Init DInputs and DWeights (but I would not need to expose them as I will do weights adjustments right here)
	// 2. Do valid and full cross-correlation
	// 3. adjust params

	inputSampleCount, _ := layer.Inputs.Dims()

	// derivatives with respect to input
	// we will use layer.DInputs directly, but it will require to write some index offet to write
	layer.DInputs = *mat.NewDense(inputSampleCount, layer.InputShape.TotalSize(), nil)
	layer.DInputs.Zero()

	// derivatives with respect to kernel values
	// will need to adjust Kernels -> using the same data structure

	// layer.dKernels should have one more dimention equal to number of samples
	// adjust params right inside `for k in 0..<inputSampleCount` loop
	layer.dKernels = make([][][]mat.Dense, inputSampleCount)
	for k := 0; k < inputSampleCount; k++ {
		layer.dKernels[k] = make([][]mat.Dense, layer.KernelShape.Depths)
		for i := 0; i < layer.KernelShape.Depths; i++ {
			layer.dKernels[k][i] = make([]mat.Dense, layer.KernelShape.InputDepths)
			for j := 0; j < layer.KernelShape.InputDepths; j++ {
				layer.dKernels[k][i][j] = *mat.DenseCopyOf(&layer.Kernels[i][j])
				layer.dKernels[k][i][j].Zero()
			}
		}
	}

	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	// going thought all input samples
	for k := 0; k < inputSampleCount; k++ {
		wg.Add(1)
		go func(layer *ConvolutionLayer, m *sync.Mutex, k int) {
			defer wg.Done()

			inputSample := ConvertSampleData(layer.Inputs.RawRowView(k), layer.InputShape)
			// dvalues for current sample
			dvalue := ConvertSampleData(dvalues.RawRowView(k), layer.OutputShape)
			// going thought all kernels within convolution layer
			for i := 0; i < layer.KernelShape.Depths; i++ {
				// going via all sub-kernels (values from 1 kernel for each input channel)
				for j := 0; j < layer.KernelShape.InputDepths; j++ {

					// valid cross-correlation between input channel j and dvalues i -> Conv::Depths * Input::Depths results
					// for input channel I have inputSample[i]
					validCorrelateResult, err := ops.Correlate2dValid(inputSample[j], dvalue[i])
					if err != nil {
						log.Fatal(err)
					}

					// full cross-correlation between i-th dvalue and [i][j] Kernel values
					// layer.DInputs =
					fullCorrelateResult, err := ops.Correlate2DFull(dvalue[i], layer.Kernels[i][j])
					if err != nil {
						log.Fatal(err)
					}

					m.Lock()
					layer.dKernels[k][i][j] = validCorrelateResult
					// Now the question how to write stuff  into DInputs
					// as layer.Dinputs is 2D array where one row will contain all the values
					// one row has shape on InputShape (as it corresponds to input data)
					// We need to sum fullCorrelateResult into dinputs[j] (so conv depths number of times for each input data channel)
					// To do this, we need a way to calculate a slices from layer.DInputs.Row(j)
					data := fullCorrelateResult.RawMatrix().Data
					startI := j * layer.InputShape.Width * layer.InputShape.Height
					// Sum dinput values according to input channel
					for ii := 0; ii < len(data); ii++ {
						idx := startI + ii
						v := layer.DInputs.At(k, idx)
						layer.DInputs.Set(k, idx, v+data[i])
					}
					m.Unlock()
				}
			}
		}(layer, &m, k)
	}
	wg.Wait()
}

func (layer *ConvolutionLayer) UpdateParams(learningRate float64) {
	// going via all used samples during Backward
	for k := 0; k < len(layer.dKernels); k++ {
		dKernelValues := layer.dKernels[k]

		for i := 0; i < len(layer.Kernels); i++ {
			for j := 0; j < len(layer.Kernels[i]); j++ {
				tmp := mat.DenseCopyOf(&dKernelValues[i][j])
				// adjusting derivatives by current learningRate
				tmp.Scale(learningRate, &dKernelValues[i][j])
				// adjusting Kernel values
				layer.Kernels[i][j].Sub(&layer.Kernels[i][j], tmp)
			}
		}

		dBiasesValues := layer.dvalues.RawRowView(k)
		for bI := 0; bI < len(layer.Biases); bI++ {
			startI := bI * layer.OutputShape.Height * layer.OutputShape.Width
			biasData := layer.Biases[bI].RawMatrix()
			for ii := 0; ii < len(biasData.Data); ii++ {
				// TODO: check if values are actually changed in layer.Biases
				biasData.Data[ii] = biasData.Data[ii] - dBiasesValues[startI+ii]*learningRate
			}
			layer.Biases[bI].SetRawMatrix(biasData)
		}
	}
}

func (layer *ConvolutionLayer) Name() string {
	return "ConvolutionLayer"
}

func (layer *ConvolutionLayer) GetOutput() *mat.Dense {
	// It is going to be a responsibility of next layer to extract data correctly
	// but it can be solved by providing InputShape struct into the next layer (if needed)
	// in case of Dense Layer we will jsut use all values from convolution as input data
	// e.g.,  with images + dense layers every pixel value was an input
	// now it is going to be an array of convolutions flatten into 1D array

	return &layer.Output
}

func (layer *ConvolutionLayer) GetDInputs() *mat.Dense {
	return &layer.DInputs
}

// Utils

// flattens all Biases into 1D array
// used to copy these values into initial Output row
func (layer *ConvolutionLayer) AllBiases() []float64 {
	all := make([]float64, 0)
	for i := 0; i < len(layer.Biases); i++ {
		all = append(all, layer.Biases[i].RawMatrix().Data...)
	}
	return all
}
