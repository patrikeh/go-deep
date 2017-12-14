# go-deep
Feed forward/backpropagation neural network implementation. Currently supports:
- Bias nodes
- L2 regularization
- Modular activation functions (sigmoid, hyperbolic, ReLU)
- Classification modes: Regression (linear output), Multiclass (softmax output)
- Momentum
- Cross-validated training

Networks are modeled as a set of neurons connected through synapses. Consequently not the fastest implementation, but hopefully an intuitive one.

Todo:
- Dropout
- Minibatches
- Other learning techniques

## Usage
Define some data...
```go
var data = []Example{
	{[]float64{2.7810836, 2.550537003}, []float64{0}},
	{[]float64{1.465489372, 2.362125076}, []float64{0}},
	{[]float64{3.396561688, 4.400293529}, []float64{0}},
	{[]float64{1.38807019, 1.850220317}, []float64{0}},
	{[]float64{7.627531214, 2.759262235}, []float64{1}},
	{[]float64{5.332441248, 2.088626775}, []float64{1}},
	{[]float64{6.922596716, 1.77106367}, []float64{1}},
	{[]float64{8.675418651, -0.242068655}, []float64{1}},
}
```

Create a network with two hidden layers of size 2 and 2 respectively:
```go
n := deep.NewNeural(&deep.Config{
	/* Input dimensionality */
	Inputs: 2,
	/* Two hidden layers consisting of two neurons each, and a single output */
	Layout: []int{2, 2, 1},
	/* Activation functions, available options are {deep.Sigmoid, deep.Tanh, deep.ReLU, deep.Linear} */
	Activation: deep.Sigmoid,
	/* Determines output layer activation: {ModeDefault, ModeRegression, ModeMulti}. 
	In the case of ModeRegression, linear outputs are used. 
	In the case of ModeMulti, a softmax output layer is applied.
	Default applies the activation defined above as per usual.*/
	Mode: ModeDefault,
	/* Weight initializers: {deep.NewNormal(stdDev, mean), deep.NewUniform(stdDev, mean)} */
	Weight: deep.NewNormal(1.0, 0.0),
	/* Error metric in cross validated training */
	Error: deep.MSE,
	/* Bias node constant - 0 disables */
	Bias: 1,
})
```
Train!
```go
n.Train(data, 1000, 0.5, 0, 0.1) // data, iterations, learning rate, regularization, momentum
```
Or with cross-validation, printing error at every 10:th epoch:
```go
training, heldout := data.Split(0.5)
n.TrainWithCrossValidation(training, heldout, 1000, 10, 0.5, 0, 0.1)
```
And make some predictions:
```go
n.Feed(data[i].Input[0]) => [0.0058055785217672636]
n.Feed(data[i].Input[5]) => [0.9936341906634203]
```

## Examples
See ```train_test.go``` for a variety of toy examples of regression, multi-class classification, binary classification, etc.

See ```examples/``` for realistic examples:

| Dataset | Topology | Epochs | Accuracy |
| --- | --- | --- | --- |
| wines | [5 5] | 10000 | ~96% |
| mnist | [100] | 50 | ~94% |
