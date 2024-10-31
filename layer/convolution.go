package layer

import (
	"log"
	"main/ops"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type InputShape struct {
	Depths int
	Height int
	Width  int
}

func (shape InputShape) TotalSize() int {
	return shape.Depths * shape.Height * shape.Width
}

type KernelShape struct {
	// defiens the total number of kernels
	Depths int
	// defines depths of kernel itself. E.g., for Grayscale input InputDepths == 1 while for RGB InputDepths == 3
	InputDepths int
	// kernelSize
	Height int
	// kernelSize
	Width int
}

func (shape KernelShape) TotalSize() int {
	return shape.Depths * shape.InputDepths * shape.Height * shape.Width
}

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
}

func (layer *ConvolutionLayer) Initialization(inputShape InputShape, convolutionDepths int, kernelSize int) {
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
		InputDepths: layer.KernelShape.InputDepths,
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
					layer.Kernels[i][j].Set(ii, jj, rand.NormFloat64()*0.01)
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
				layer.Biases[i].Set(ii, jj, rand.NormFloat64()*0.01)
			}
		}
	}
}

func (layer *ConvolutionLayer) Forward(inputs *mat.Dense, isTraining bool) {
	layer.Inputs = *inputs
	inputSampleCount, _ := inputs.Dims()

	// validate input shape
	row := inputs.RawMatrix().Data
	if len(row) != layer.InputShape.TotalSize() {
		log.Fatalf("Unexpected input size: %v with InputShape: %v", len(row), layer.InputShape)
	}

	// now we need to create Output
	// Rows -> number of input samples. we can get from inputs.Dims() -> R
	// Cols -> it is defined by layer.OutputShape.TotalSize()

	layer.Output = *mat.NewDense(inputSampleCount, layer.OutputShape.TotalSize(), nil)
	layer.Output.Zero()

	allBiases := layer.allBiases()
	if len(allBiases) != layer.OutputShape.TotalSize() {
		log.Fatalf("Unexpected length of biases: %v with OutputShape: %v", len(allBiases), layer.OutputShape)
	}
	for i := 0; i < inputSampleCount; i++ {
		// copy all biases values into Output
		// then we will start adding results of convolution into output that already have biases
		layer.Output.SetRow(i, allBiases)
	}

	// going thought all input samples
	for k := 0; k < inputSampleCount; k++ {
		// for convenience raw data from one sample from inputs is converted according to InputShape
		inputSample := layer.ConvertSampleData(inputs.RawRowView(k), layer.InputShape)

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
				for k := 0; k < len(rawConvResult); k++ {
					outputIdx := i*layer.OutputShape.Height*layer.OutputShape.Width + k
					layer.Output.Set(i, outputIdx, rawConvResult[k])
				}
			}
		}
	}

	// Output shape will be number of input samples * layer.OutputShape.TotalSize()
}

// dvalues comes form max pooling (well, via relu)
func (layer *ConvolutionLayer) Backward(dvalues *mat.Dense) {
	// okay, now I can start building Backward pass
	// 1. Init DInputs and DWeights (but I would not need to expose them as I will do weights adjustments right here)
	// 2. Do valid and full cross-correlation
	// 3. adjust params

	inputSampleCount, _ := layer.Inputs.Dims()

	// derivatives with respect to input
	// we will use layer.DInputs directly, but it will require to write some index offet to write
	layer.DInputs = *mat.NewDense(inputSampleCount, layer.OutputShape.TotalSize(), nil)
	layer.DInputs.Zero()

	// derivatives with respect to kernel values
	// will need to adjust Kernels -> using the same data structure
	DKernels := make([][]mat.Dense, layer.KernelShape.Depths)
	for i := 0; i < layer.KernelShape.Depths; i++ {
		DKernels[i] = make([]mat.Dense, layer.KernelShape.InputDepths)
		for j := 0; j < layer.KernelShape.InputDepths; j++ {
			DKernels[i][j] = *mat.DenseCopyOf(&layer.Kernels[i][j])
			DKernels[i][j].Zero()
		}
	}

	// going thought all input samples
	for k := 0; k < inputSampleCount; k++ {
		inputSample := layer.ConvertSampleData(layer.Inputs.RawRowView(k), layer.InputShape)
		// dvalues for current sample
		dvalue := layer.ConvertSampleData(dvalues.RawRowView(k), layer.OutputShape)
		// going thought all kernels within convolution layer
		for i := 0; i < layer.KernelShape.Depths; i++ {
			// going via all sub-kernels (values from 1 kernel for each input channel)
			for j := 0; j < layer.KernelShape.InputDepths; j++ {

				// valid cross-correlation between input channel j and dvalues i -> Conv::Depths * Input::Depths results
				// for input channel I have inputSample[i]
				// DKernels[i][j] =
				ops.Correlate2dValid(inputSample[j], dvalue[i])

				// full cross-correlation between i-th dvalue and [i][j] Kernel values
				// layer.DInputs =
				ops.Correlate2DFull(dvalue[i], layer.Kernels[i][j])

				// TODO: write the result into DInputs and DKernels
			}
		}

		// TODO:
		// get current learning rate
		// adjust kernels
		// adjust biases
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
	panic(1)
}

// Utils

// flattens all Biases into 1D array
// used to copy these values into initial Output row
func (layer *ConvolutionLayer) allBiases() []float64 {
	all := make([]float64, 0)
	for i := 0; i < len(layer.Biases); i++ {
		all = append(all, layer.Biases[i].RawMatrix().Data...)
	}
	return all
}

// takes raw data from one sample for inputs and slices it according to InputShape
// e.g., Grayscake will return one mat.Dense
// RGB - len == 3
func (layer *ConvolutionLayer) ConvertSampleData(inputRawData []float64, shape InputShape) []mat.Dense {
	// TODO: add test for this func
	data := make([]mat.Dense, shape.Depths)
	for k := 0; k < shape.Depths; k++ {
		slice := inputRawData[k*shape.Height*shape.Width : (k+1)*shape.Height*shape.Width]
		data[k] = *mat.NewDense(shape.Height, shape.Width, slice)
	}

	return data
}
