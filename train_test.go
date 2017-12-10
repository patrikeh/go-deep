package deep

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_BoundedRegression(t *testing.T) {
	rand.Seed(0)

	squares := Examples{}
	for i := 0.0; i < 1; i += 0.01 {
		squares = append(squares, Example{Input: []float64{i}, Response: []float64{math.Pow(i, 2)}})
	}
	n := NewNeural(&Config{
		Inputs:     1,
		Layout:     []int{4, 4, 1},
		Activation: Sigmoid,
		Weight:     Random,
		Error:      MSE,
		Bias:       1,
	})

	n.Train(squares, 1000, 0.1, 0)

	tests := []float64{0.0, 0.1, 0.5, 0.75, 0.9}
	for _, x := range tests {
		assert.InEpsilon(t, math.Pow(x, 2)+1, n.Forward([]float64{x})[0]+1, 0.1)
	}
}

func Test_RegressionLinearOuts(t *testing.T) {
	rand.Seed(0)
	squares := Examples{}
	for i := 0.0; i < 100.0; i++ {
		squares = append(squares, Example{Input: []float64{i}, Response: []float64{math.Sqrt(i)}})
	}
	n := NewNeural(&Config{
		Inputs:        1,
		Layout:        []int{3, 3, 1},
		Activation:    ReLU,
		OutActivation: &Linear,
		Weight:        NewNormal(1, 0),
		Bias:          1,
	})

	n.Train(squares, 3000, 0.0001, 0.0001)

	for i := 0; i < 20; i++ {
		x := float64(rand.Intn(100))
		fmt.Printf("want: %+v have: %+v\n", math.Sqrt(x), n.Forward([]float64{x}))
		assert.InEpsilon(t, math.Sqrt(x), n.Forward([]float64{x})[0], 0.1)
	}
}

func Test_Training(t *testing.T) {
	rand.Seed(0)

	data := Examples{
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

	v := n.Forward([]float64{0})
	assert.InEpsilon(t, 1, 1+v[0], 0.1)
	v = n.Forward([]float64{5})
	assert.InEpsilon(t, 1.0, v[0], 0.1)
}

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

func Test_Prediction(t *testing.T) {
	rand.Seed(0)

	n := NewNeural(&Config{
		Inputs:     2,
		Layout:     []int{2, 2, 1},
		Activation: Sigmoid,
		Weight:     Random,
		Bias:       1,
	})

	n.Train(data, 5000, 0.5, 0)

	for _, d := range data {
		assert.InEpsilon(t, n.Forward(d.Input)[0]+1, d.Response[0]+1, 0.1)
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

	n.TrainWithCrossValidation(data, data, 1000, 10, 0.5, 0.0001)

	for _, d := range data {
		assert.InEpsilon(t, n.Forward(d.Input)[0]+1, d.Response[0]+1, 0.1)
		assert.InEpsilon(t, 1, n.CrossValidate(data)+1, 0.01)
	}
}
