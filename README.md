# go-deep
Another not so feature-complete neural network implementation. Currently supports some small extras that I was not able to find in other packages:
- Bias nodes
- L2 regularization
- Modular activation functions (currently hyperbolic, sigmoid, leaky ReLU)
- Cross-validated training

## Usage
Define some data...
```
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
```

Create a network with two hidden layers of size 2 and 2 respectively:
```
n := deep.NewNeural(&deep.Config{
	Inputs:     2,                                      // Input dimensionality
	Layout:     []int{2, 2, 1},                         // 2x hidden layers with 2 nodes each, 1 output
	Activation: {deep.Sigmoid, deep.Tanh, deep.ReLU},   // Activation function
	Weight:     {deep.Random, deep.Normal},             // Weight initializers
	Error:      deep.MSE,                               // Loss function
	Bias:       1,                                      // Bias constant (0 disables)
})
```
Train!
```
n.Train(data, 1000, 0.5, 0) // Data, iterations, learning rate, L2 regularization parameter (gamma)
```
Or with cross-validation:
```
training, heldout := data.Split(0.5)
n.TrainWithCrossValidation(training, heldout, 1000, 10, 0.5, 0)
```
And make some predictions:
```
n.Feed(data[i].Input[0]) => [0.9936341906634203]
n.Feed(data[i].Input[5]) => [0.0058055785217672636]
```
