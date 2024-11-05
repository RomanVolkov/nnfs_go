package marshaling

import (
	"errors"
	"main/layer"
	"testing"
)

func TestConvolutionMarshaling(t *testing.T) {
	l := layer.ConvolutionLayer{}
	inputShape := layer.InputShape{Depths: 1, Width: 20, Height: 20}

	l.Initialization(inputShape, 2, 3)

	wrap := ConvolutionWrapper{ConvolutionLayer: l}

	d, err := wrap.MarshalJSON()
	if err != nil {
		t.Error(err)
	}

	t.Log(string(d))

	loadedWrap := ConvolutionWrapper{}
	err = loadedWrap.UnmarshalJSON(d)
	if err != nil {
		t.Error(err)
	}

	if l.InputShape != loadedWrap.InputShape {
		t.Error(errors.New("mismatch in InputShape"))
	}

	if l.OutputShape != loadedWrap.OutputShape {
		t.Error(errors.New("mismatch in OutputShape"))
	}

	if l.KernelShape != loadedWrap.KernelShape {
		t.Error(errors.New("mismatch in KernelShape"))
	}

	if l.Depths != loadedWrap.Depths {
		t.Error(errors.New("mismatch in Depths"))
	}

	if !IsEqual(l.Kernels[0][0].RawMatrix().Data, loadedWrap.Kernels[0][0].RawMatrix().Data) {
		t.Error(errors.New("mismatch in Kernels"))
	}

	if !IsEqual(l.Biases[0].RawMatrix().Data, loadedWrap.Biases[0].RawMatrix().Data) {
		t.Error(errors.New("mismatch in Biases"))
	}
}
