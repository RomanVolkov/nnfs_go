package layer

import (
	"log"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type MaxPoolingLayer struct {
	// size of the pool for max pooling
	PoolSize int `json:"pool_size"`
	// size of output from Conv layer
	InputShape InputShape `json:"input_shape"`
	// pooling output size
	OutputShape InputShape `json:"output_shape"`

	Output mat.Dense

	DInputs mat.Dense

	// input data from Forward to be used in Backward
	inputs  mat.Dense
	dvalues mat.Dense
}

func (layer *MaxPoolingLayer) Initialization(inputShape InputShape, poolSize int) *MaxPoolingLayer {
	layer.PoolSize = poolSize
	layer.InputShape = inputShape

	layer.OutputShape = InputShape{
		Depths: inputShape.Depths,
		Width:  inputShape.Width / poolSize,
		Height: inputShape.Height / poolSize,
	}

	return layer
}

func (layer *MaxPoolingLayer) LoadFromParams(poolSize int, inputShape InputShape, outputShape InputShape) {
	layer.PoolSize = poolSize
	layer.InputShape = inputShape
	layer.OutputShape = outputShape
}

// inputs comes from ConvLayer
// output size from ConvLayer; number of kernel from ConvLayer
func (layer *MaxPoolingLayer) Forward(inputs *mat.Dense, isTraining bool) {
	layer.inputs = *mat.DenseCopyOf(inputs)

	inputSampleCount, _ := inputs.Dims()

	// validate input shape
	row := inputs.RawRowView(0)
	if len(row) != layer.InputShape.TotalSize() {
		log.Fatalf("Unexpected input size: %v with InputShape: %v", len(row), layer.InputShape)
	}

	layer.Output = *mat.NewDense(inputSampleCount, layer.OutputShape.TotalSize(), nil)
	layer.Output.Zero()

	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	// going thought all input samples
	for k := 0; k < inputSampleCount; k++ {
		wg.Add(1)
		go func(layer *MaxPoolingLayer, m *sync.Mutex, k int) {
			wg.Done()
			inputSample := ConvertSampleData(layer.inputs.RawRowView(k), layer.InputShape)
			outputSample := make([]float64, layer.OutputShape.TotalSize())

			// going through input channels from sample
			// OutputShape.Depths == InputShape.Depths
			for c := 0; c < layer.OutputShape.Depths; c++ {
				for i := 0; i < layer.OutputShape.Height; i++ {
					for j := 0; j < layer.OutputShape.Width; j++ {
						startI := i * layer.PoolSize
						startJ := j * layer.PoolSize
						endI := startI + layer.PoolSize
						endJ := startJ + layer.PoolSize

						patch := inputSample[c].Slice(startI, endI, startJ, endJ)
						value := MaxValue(patch)

						idx := c*layer.OutputShape.Width*layer.OutputShape.Height + i*layer.OutputShape.Height + j
						outputSample[idx] = value
					}
				}
			}

			layer.Output.SetRow(k, outputSample)
		}(layer, &m, k)
	}
	wg.Wait()
}

func (layer *MaxPoolingLayer) Backward(dvalues *mat.Dense) {
	layer.dvalues = *mat.DenseCopyOf(dvalues)
	inputSampleCount, _ := layer.inputs.Dims()

	layer.DInputs = *mat.NewDense(inputSampleCount, layer.InputShape.TotalSize(), nil)
	layer.DInputs.Zero()

	wg := sync.WaitGroup{}
	m := sync.Mutex{}

	// going thought all samples
	for k := 0; k < inputSampleCount; k++ {
		wg.Add(1)
		go func(layer *MaxPoolingLayer, m *sync.Mutex, k int) {
			defer wg.Done()

			inputSample := ConvertSampleData(layer.inputs.RawRowView(k), layer.InputShape)
			dvalueSample := ConvertSampleData(dvalues.RawRowView(k), layer.OutputShape)
			outputDInput := make([]float64, layer.InputShape.TotalSize())
			// going through input channels from sample
			// OutputShape.Depths == InputShape.Depths
			for c := 0; c < layer.OutputShape.Depths; c++ {
				for i := 0; i < layer.OutputShape.Height; i++ {
					for j := 0; j < layer.OutputShape.Width; j++ {
						startI := i * layer.PoolSize
						startJ := j * layer.PoolSize
						endI := startI + layer.PoolSize
						endJ := startJ + layer.PoolSize

						patch := inputSample[c].Slice(startI, endI, startJ, endJ)
						value := MaxValue(patch)
						mask := mat.DenseCopyOf(patch)
						mask.Apply(func(i, j int, v float64) float64 {
							if v == value {
								return 1.
							} else {
								return 0.
							}
						}, mask)

						// set dinputs based on input dvalues and calculated mask
						for ii := startI; ii < endI; ii++ {
							for jj := startJ; jj < endJ; jj++ {
								idx := c*layer.InputShape.Height*layer.InputShape.Width + ii*layer.InputShape.Height + jj
								outputDInput[idx] = dvalueSample[c].At(i, j) * mask.At(ii-startI, jj-startJ)
							}
						}
					}
				}
			}

			m.Lock()
			layer.DInputs.SetRow(k, outputDInput)
			m.Unlock()

		}(layer, &m, k)
	}
	wg.Wait()
}

func (layer *MaxPoolingLayer) Name() string {
	return "MaxPoolingLayer"
}

func (layer *MaxPoolingLayer) GetOutput() *mat.Dense {
	return &layer.Output
}

func (layer *MaxPoolingLayer) GetDInputs() *mat.Dense {
	return &layer.DInputs
}
