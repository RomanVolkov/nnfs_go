package marshaling

import (
	"encoding/json"

	"gonum.org/v1/gonum/mat"
)

type DenseWrapper struct {
	mat.Dense
}

type denseWrapperData struct {
	Values []float64 `json:"values"`
	Rows   int       `json:"rows"`
	Cols   int       `json:"cols"`
}

func (value DenseWrapper) MarshalJSON() ([]byte, error) {
	values := value.Dense.RawMatrix().Data
	rows, cols := value.Dense.Dims()

	wrap := denseWrapperData{Values: values, Rows: rows, Cols: cols}

	return json.Marshal(wrap)
}

func (value *DenseWrapper) UnmarshalJSON(data []byte) error {
	wrap := denseWrapperData{}
	err := json.Unmarshal(data, &wrap)
	if err != nil {
		return err
	}
	value.Dense = *mat.NewDense(wrap.Rows, wrap.Cols, wrap.Values)
	return nil
}
