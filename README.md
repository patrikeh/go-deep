# go-deep
Feed forward/backpropagation neural network implementation. Currently supports:

- Modular activation functions (sigmoid, hyperbolic, ReLU)
- Classification modes: regression, multi-class, multi-label, binary
- Bias
- Cross-validated training
- Modular solvers (currently supports SGD, SGD with momentum/nesterov, Adam)
Networks are modeled as a set of neurons connected through synapses. Consequently not the fastest implementation, but hopefully an intuitive one.

Todo:
- Dropout
- Batch normalization

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
	/* Determines output layer activation: 
	{ModeBinary, ModeRegression, ModeMultiClass, ModeMultiLabel}. 
	In the case of ModeRegression, linear outputs are used. 
	In the case of ModeMulticlass, a softmax output layer is applied.
	Default applies the activation defined above as per usual.*/
	Mode: ModeDefault,
	/* Weight initializers: {deep.NewNormal(stdDev, mean), deep.NewUniform(stdDev, mean)} */
	Weight: deep.NewNormal(1.0, 0.0),
	/* Apply bias */
	Bias: true,
})
```
Train:
```go
// params: learning rate, momentum, nesterov
optimizer := training.NewSGD(0.05, 0.1, true)
// params: optimizer, verbosity (print stats at every 50th iteration)
trainer := training.NewTrainer(optimizer, 50)

training, heldout := data.Split(0.5)
trainer.Train(n, training, heldout, 1000) // training, validation, iterations
```
resulting in:
```
Epochs        Elapsed       Error         
---           ---           ---           
5             12.938µs      0.36438       
10            125.691µs     0.02261       
15            177.194µs     0.00404       
...     
1000          10.703839ms   0.00000       
```
Finally, make some predictions:
```go
n.Predict(data[0].Input) => [0.0058055785217672636]
n.Predict(data[5].Input) => [0.9936341906634203]
```

Alternatively, batch training can be performed in parallell:
```go
optimizer := NewAdam(0.001, 0.9, 0.999, 1e-8)
// params: optimizer, verbosity (print info at every n:th iteration), batch-size, number of workers
trainer := training.NewBatchTrainer(optimizer, 1, 200, 4)

training, heldout := data.Split(0.75)
trainer.Train(n, training, heldout, 1000) // training, validation, iterations
```

## Examples
See ```training/trainer_test.go``` for a variety of toy examples of regression, multi-class classification, binary classification, etc.

See ```examples/``` for realistic examples:

| Dataset | Topology | Epochs | Accuracy |
| --- | --- | --- | --- |
| wines | [5 5] | 10000 | ~98% |
| mnist | [50] | 25 | ~97% |
