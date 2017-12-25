package training

import (
	"math"
	"math/rand"
	"testing"

	deep "github.com/patrikeh/go-deep"
	"github.com/stretchr/testify/assert"
)

func Test_BoundedRegression(t *testing.T) {
	rand.Seed(0)

	squares := Examples{}
	for i := 0.0; i < 1; i += 0.01 {
		squares = append(squares, Example{Input: []float64{i}, Response: []float64{math.Pow(i, 2)}})
	}
	n := deep.NewNeural(&deep.Config{
		Inputs:     1,
		Layout:     []int{4, 4, 1},
		Activation: deep.ActivationSigmoid,
		Weight:     deep.NewUniform(0.5, 0),
		Error:      deep.MSE,
		Bias:       0,
	})

	trainer := NewTrainer(0.1, 0, 0.9, 0)
	trainer.Train(n, squares, nil, 1000)

	tests := []float64{0.0, 0.1, 0.5, 0.75, 0.9}
	for _, x := range tests {
		assert.InEpsilon(t, math.Pow(x, 2)+1, n.Predict([]float64{x})[0]+1, 0.1)
	}
}

func Test_RegressionLinearOuts(t *testing.T) {
	rand.Seed(0)
	squares := Examples{}
	for i := 0.0; i < 100.0; i++ {
		squares = append(squares, Example{Input: []float64{i}, Response: []float64{math.Sqrt(i)}})
	}
	squares.Shuffle()
	n := deep.NewNeural(&deep.Config{
		Inputs:     1,
		Layout:     []int{3, 3, 1},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeRegression,
		Weight:     deep.NewNormal(0.5, 0.5),
		Bias:       1,
	})

	trainer := NewTrainer(0.001, 0.0, 1, 0)
	trainer.Train(n, squares, nil, 20000)

	for i := 0; i < 20; i++ {
		x := float64(rand.Intn(99) + 1)
		assert.InEpsilon(t, math.Sqrt(x)+1, n.Predict([]float64{x})[0]+1, 0.1)
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

	n := deep.NewNeural(&deep.Config{
		Inputs:     1,
		Layout:     []int{5, 1},
		Activation: deep.ActivationSigmoid,
		Weight:     deep.NewUniform(0.5, 0),
		Bias:       1,
	})

	trainer := NewTrainer(0.5, 0, 0.1, 0)
	trainer.Train(n, data, nil, 1000)

	v := n.Predict([]float64{0})
	assert.InEpsilon(t, 1, 1+v[0], 0.1)
	v = n.Predict([]float64{5})
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

	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{2, 2, 1},
		Activation: deep.ActivationSigmoid,
		Weight:     deep.NewUniform(0.5, 0),
		Bias:       1,
	})

	trainer := NewTrainer(0.5, 0, 0.1, 0)

	trainer.Train(n, data, nil, 5000)

	for _, d := range data {
		assert.InEpsilon(t, n.Predict(d.Input)[0]+1, d.Response[0]+1, 0.1)
	}
}

func Test_CrossVal(t *testing.T) {
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{1, 1},
		Activation: deep.ActivationTanh,
		Weight:     deep.NewUniform(0.5, 0),
		Error:      deep.MSE,
		Bias:       1,
	})

	trainer := NewTrainer(0.5, 0.0001, 0.1, 0)
	trainer.Train(n, data, data, 1000)

	for _, d := range data {
		assert.InEpsilon(t, n.Predict(d.Input)[0]+1, d.Response[0]+1, 0.1)
		assert.InEpsilon(t, 1, trainer.CrossValidate(n, data)+1, 0.01)
	}
}

func Test_MultiClass(t *testing.T) {
	var data = []Example{
		{[]float64{2.7810836, 2.550537003}, []float64{1, 0}},
		{[]float64{1.465489372, 2.362125076}, []float64{1, 0}},
		{[]float64{3.396561688, 4.400293529}, []float64{1, 0}},
		{[]float64{1.38807019, 1.850220317}, []float64{1, 0}},
		{[]float64{3.06407232, 3.005305973}, []float64{1, 0}},
		{[]float64{7.627531214, 2.759262235}, []float64{0, 1}},
		{[]float64{5.332441248, 2.088626775}, []float64{0, 1}},
		{[]float64{6.922596716, 1.77106367}, []float64{0, 1}},
		{[]float64{8.675418651, -0.242068655}, []float64{0, 1}},
		{[]float64{7.673756466, 3.508563011}, []float64{0, 1}},
	}

	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{2, 2},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMulti,
		Weight:     deep.NewUniform(0.1, 0),
		Error:      deep.MSE,
		Bias:       1,
	})

	trainer := NewTrainer(0.01, 0.0001, 0.1, 0)
	trainer.Train(n, data, data, 1000)

	for _, d := range data {
		est := n.Predict(d.Input)
		assert.InEpsilon(t, 1.0, deep.Sum(est), 0.00001)
		if d.Response[0] == 1.0 {
			assert.InEpsilon(t, n.Predict(d.Input)[0]+1, d.Response[0]+1, 0.1)
		} else {
			assert.InEpsilon(t, n.Predict(d.Input)[1]+1, d.Response[1]+1, 0.1)
		}
		assert.InEpsilon(t, 1, trainer.CrossValidate(n, data)+1, 0.01)
	}

}

func Test_xor(t *testing.T) {
	rand.Seed(0)
	n := deep.NewNeural(&deep.Config{
		Inputs:     2,
		Layout:     []int{2, 1}, // Should be sufficient for modeling (AND+OR)
		Activation: deep.ActivationSigmoid,
		Weight:     deep.NewUniform(.25, 0),
		Bias:       1,
	})
	permutations := Examples{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}

	trainer := NewTrainer(0.9, 0.0001, 0.1, 0)
	trainer.Train(n, permutations, nil, 1000)

	for _, perm := range permutations {
		assert.InEpsilon(t, n.Predict(perm.Input)[0]+1, perm.Response[0]+1, 0.2)
	}
}
