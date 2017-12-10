package deep

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

var data = []Example{
	{[]float64{2.7810836, 2.550537003}, []float64{0}},
	{[]float64{1.465489372, 2.362125076}, []float64{0}},
	{[]float64{3.396561688, 4.400293529}, []float64{0}},
	{[]float64{1.38807019, 1.850220317}, []float64{0}},
	{[]float64{3.06407232, 3.005305973}, []float64{0}},
	{[]float64{7.627531214, 2.759262235}, []float64{1}},
	{[]float64{5.332441248, 2.088626775}, []float64{1}},
	{[]float64{6.922596716, 1.77106367}, []float64{1}},
	{[]float64{8.675418651, -0.242068655}, []float64{1}},
	{[]float64{7.673756466, 3.508563011}, []float64{1}},
}

func Test_LinearOuts(t *testing.T) {
	squares := Examples{}
	for i := 0.0; i < 15; i++ {
		squares = append(squares, Example{Input: []float64{i}, Response: []float64{math.Pow(i, 2)}})
	}
	n := NewNeural(&Config{
		Inputs:        1,
		Layout:        []int{2, 2, 1},
		Activation:    Sigmoid,
		OutActivation: &Linear,
		Weight:        Random,
		Bias:          0,
	})
	n.Train(squares, 10, 0.1, 0.0)
	fmt.Printf("%+v\n", n.Feed([]float64{1}))
	fmt.Printf("%+v\n", n.Feed([]float64{10}))

}
func Test_Training(t *testing.T) {
	data := []Example{
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{0}, []float64{0}},
		Example{[]float64{5}, []float64{1}},
		Example{[]float64{5}, []float64{1}},
	}

	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{1, 1},
		Activation: Sigmoid,
		Weight:     Random,
		Bias:       1,
	})

	for i := 0; i < 1000; i++ {
		for _, data := range data {
			n.Learn(data, 0.5, 0)
		}
	}

	v := n.Feed([]float64{0})
	assert.InEpsilon(t, 1, 1+v[0], 0.1)
	v = n.Feed([]float64{5})
	assert.InEpsilon(t, 1.0, v[0], 0.1)
}

func Test_Prediction(t *testing.T) {

	n := NewNeural(&Config{
		Inputs:     2,
		Layout:     []int{2, 2, 1},
		Activation: Sigmoid,
		Weight:     Random,
		Bias:       1,
	})

	n.Train(data, 5000, 0.5, 0)

	for _, d := range data {
		fmt.Printf("%+v\n", n.Feed(d.Input)[0])
		assert.InEpsilon(t, n.Feed(d.Input)[0]+1, d.Response[0]+1, 0.1)
	}
}

func Test_CrossVal(t *testing.T) {

	n := NewNeural(&Config{
		Inputs:     2,
		Layout:     []int{1, 1},
		Activation: Tanh,
		Weight:     Random,
		Error:      MSE,
		Bias:       1,
	})

	n.TrainWithCrossValidation(data, data, 1000, 10, 0.5, 0)

	for _, d := range data {
		assert.InEpsilon(t, n.Feed(d.Input)[0]+1, d.Response[0]+1, 0.1)
		assert.InEpsilon(t, 1, n.CrossValidate(data)+1, 0.01)
	}
}
