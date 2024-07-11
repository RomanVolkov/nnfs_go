package optimizations

import (
	"main/activation"
	"main/layer"
	"main/loss"

	"gonum.org/v1/gonum/mat"
)

// Activation

type OptimizedSoftmaxActivation struct {
	activation             activation.SoftmaxActivation
	backwardImplementation *ActivationSoftmaxLossCategorialCrossentropy
}

func (a *OptimizedSoftmaxActivation) Name() string {
	return "Optimzied Softmax Activation"
}

func (activation *OptimizedSoftmaxActivation) Forward(inputs *mat.Dense, isTraining bool) {
	activation.activation.Forward(inputs, isTraining)
}

func (activation *OptimizedSoftmaxActivation) Backward(dvalues *mat.Dense) {}

func (a *OptimizedSoftmaxActivation) GetOutput() *mat.Dense {
	return a.activation.GetOutput()
}

func (a *OptimizedSoftmaxActivation) GetDInputs() *mat.Dense {
	return &a.backwardImplementation.DInputs
}

func (a *OptimizedSoftmaxActivation) Predictions(outputs *mat.Dense) mat.Dense {
	return *mat.DenseCopyOf(outputs)
}

// Loss

type OptimizedCategoricalCrossentropyLoss struct {
	loss                   loss.CategoricalCrossentropyLoss
	backwardImplementation *ActivationSoftmaxLossCategorialCrossentropy
}

func (loss *OptimizedCategoricalCrossentropyLoss) Name() string {
	return "Optimized Categorial Crossentropy Loss"
}

func (loss *OptimizedCategoricalCrossentropyLoss) Forward(prediction *mat.Dense, target *mat.Dense) []float64 {
	return loss.loss.Forward(prediction, target)
}

func (loss *OptimizedCategoricalCrossentropyLoss) Backward(dvalues *mat.Dense, target *mat.Dense) {
	loss.backwardImplementation.Backward(dvalues, target)
}

func (loss *OptimizedCategoricalCrossentropyLoss) GetDInputs() *mat.Dense {
	return nil
}

func (loss *OptimizedCategoricalCrossentropyLoss) SetLayers(layers []*layer.Layer) {
	loss.loss.SetLayers(layers)
}

func (loss *OptimizedCategoricalCrossentropyLoss) RegularizationLoss() float64 {
	return loss.loss.RegularizationLoss()
}

// Factory

func MakeOptimizedCategorialCrossentropy() (OptimizedSoftmaxActivation, OptimizedCategoricalCrossentropyLoss) {
	a := OptimizedSoftmaxActivation{}
	a.activation = activation.SoftmaxActivation{}

	l := OptimizedCategoricalCrossentropyLoss{}
	l.loss = loss.CategoricalCrossentropyLoss{}

	a.backwardImplementation = &ActivationSoftmaxLossCategorialCrossentropy{}
	l.backwardImplementation = a.backwardImplementation

	return a, l
}

// Optimized backward implementation

type ActivationSoftmaxLossCategorialCrossentropy struct {
	DInputs mat.Dense
}

func (a *ActivationSoftmaxLossCategorialCrossentropy) Backward(dvalues *mat.Dense, target *mat.Dense) {
	_, c := target.Dims()
	if c != 1 {
		panic("target should have only 1 column")
	}
	samplesCount, _ := dvalues.Dims()
	a.DInputs = *mat.DenseCopyOf(dvalues)

	for i := 0; i < samplesCount; i++ {
		value := a.DInputs.At(i, int(target.At(i, 0))) - 1.0
		a.DInputs.Set(i, int(target.At(i, 0)), value)
	}

	a.DInputs.Apply(func(i, j int, v float64) float64 {
		return v / float64(samplesCount)
	}, &a.DInputs)
}
