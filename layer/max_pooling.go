package layer

import (
	"main/utils"

	"gonum.org/v1/gonum/mat"
)

// NOT USED NOW
type MaxPoolingLayer struct {
	// size of the pool for max pooling
	PoolSize int
	// size of output from Conv layer
	InputSize int
	// kernel count form Con layer
	KernelCount int

	// pooling output size
	OutputSize int

	// rows - corresponds to samples count
	// cols - contains from results of max pooling for all kernerls. OutputSize ^ 2 * KernelCount
	Output  mat.Dense
	DInputs mat.Dense

	// input data from Forward to be used in Backward
	inputs mat.Dense
}

// inputs comes from ConvLayer
// output size from ConvLayer; number of kernel from ConvLayer
func (layer *MaxPoolingLayer) Forward(inputs *mat.Dense, isTraining bool) {
	layer.inputs = *inputs
	// take the number of samles for training or inference
	rows, _ := inputs.Dims()
	// calcualte Max Pooling output size
	layer.OutputSize = layer.InputSize / layer.PoolSize
	layer.Output = *mat.NewDense(rows, layer.KernelCount*layer.OutputSize*layer.OutputSize, nil)

	// going over all samples
	for sampleI := 0; sampleI < rows; sampleI++ {
		// data that contains all outputs from Conv layer
		// for current sample
		inputData := inputs.RawRowView(sampleI)
		// allocate slice to collect all MaxPooling results from all kernetls
		maxPoolingOutput := make([]float64, layer.KernelCount*layer.OutputSize^2)
		// going over all kernels
		for kernelI := 0; kernelI < layer.KernelCount; kernelI++ {
			// declare max pooling oputput for kernel for sample
			kernelMaxPoolingData := mat.NewDense(layer.OutputSize, layer.OutputSize, nil)

			// interating over output array
			for i := 0; i < layer.OutputSize; i++ {
				for j := 0; j < layer.OutputSize; j++ {
					// global indexes from sample
					idxI := (kernelI*layer.InputSize ^ 2 + i*layer.InputSize)
					idxJ := idxI + j

					// find a patch window from input data
					startI, startJ := idxI*layer.PoolSize, idxJ*layer.PoolSize
					endI, endJ := startI+layer.PoolSize, startJ+layer.PoolSize

					poolValue := utils.MaxValue(inputData[i*layer.InputSize:(i+1)*layer.InputSize], startI, endI, startJ, endJ, layer.InputSize)
					kernelMaxPoolingData.Set(i, j, poolValue)
				}
			}

			maxPoolingOutput = append(maxPoolingOutput, kernelMaxPoolingData.RawMatrix().Data...)
		}
		layer.Output.SetRow(sampleI, maxPoolingOutput)
	}
}

func (layer *MaxPoolingLayer) Backward(dvalues *mat.Dense) {
	layer.DInputs = *mat.DenseCopyOf(&layer.inputs)
	layer.DInputs.Zero()
	rows, _ := dvalues.Dims()

	// going over all samples
	for sampleI := 0; sampleI < rows; sampleI++ {
		// data that contains all outputs from Conv layer
		// for current sample
		inputData := layer.inputs.RawRowView(sampleI)
		// going over all kernels
		for kernelI := 0; kernelI < layer.KernelCount; kernelI++ {
			// interating over output array
			for i := 0; i < layer.OutputSize; i++ {
				for j := 0; j < layer.OutputSize; j++ {
					// TODO: check indexes and InputSize vs OutputSize
					// global indexes from sample
					idxI := (kernelI*layer.InputSize ^ 2 + i*layer.InputSize)
					idxJ := idxI + j

					// find a patch window from input data
					startI, startJ := idxI*layer.PoolSize, idxJ*layer.PoolSize
					endI, endJ := startI+layer.PoolSize, startJ+layer.PoolSize

					poolValue := utils.MaxValue(inputData[i*layer.InputSize:(i+1)*layer.InputSize], startI, endI, startJ, endJ, layer.InputSize)

					for ii := startI; ii <= endI; ii++ {
						for jj := startJ; jj <= endJ; jj++ {
							dvalue := dvalues.At(sampleI, kernelI*layer.InputSize^2+i*layer.InputSize+j)
							v := 0.
							if poolValue == layer.inputs.At(sampleI, kernelI*layer.InputSize^2+ii*layer.InputSize+jj) {
								v = dvalue
							}

							layer.DInputs.Set(sampleI, kernelI*layer.InputSize^2+ii*layer.InputSize+jj, v)
						}
					}
				}
			}
		}
	}
}
