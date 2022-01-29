# go-deep

[![GoDoc](https://godoc.org/github.com/patrikeh/go-deep?status.svg)](https://godoc.org/github.com/patrikeh/go-deep)
[![Go Report Card](https://goreportcard.com/badge/github.com/patrikeh/go-deep)](https://goreportcard.com/report/github.com/patrikeh/go-deep)
[![CircleCI](https://circleci.com/gh/patrikeh/go-deep/tree/master.svg?style=svg)](https://circleci.com/gh/patrikeh/go-deep/tree/master)
[![codecov](https://codecov.io/gh/patrikeh/go-deep/branch/master/graph/badge.svg?token=fFCrxfhuL0)](https://codecov.io/gh/patrikeh/go-deep)

Feed forward/backpropagation neural network implementation. Currently supports:

- Activation functions: sigmoid, hyperbolic, ReLU
- Solvers: SGD, SGD with momentum/nesterov, Adam
- Classification modes: regression, multi-class, multi-label, binary
- Supports batch training in parallel
- Bias nodes

Networks are modeled as a set of neurons connected through synapses. No GPU computations - don't use this for any large scale applications.

## Install

```
go get -u github.com/patrikeh/go-deep
```

## Usage

Import the go-deep package

```go
import (
	"fmt"
	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)
```

Define some data...

```go
var data = training.Examples{
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
	/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
	Activation: deep.ActivationSigmoid,
	/* Determines output layer activation & loss function:
	ModeRegression: linear outputs with MSE loss
	ModeMultiClass: softmax output with Cross Entropy loss
	ModeMultiLabel: sigmoid output with Cross Entropy loss
	ModeBinary: sigmoid output with binary CE loss */
	Mode: deep.ModeBinary,
	/* Weight initializers: {deep.NewNormal(μ, σ), deep.NewUniform(μ, σ)} */
	Weight: deep.NewNormal(1.0, 0.0),
	/* Apply bias */
	Bias: true,
})
```

Train:

```go
// params: learning rate, momentum, alpha decay, nesterov
optimizer := training.NewSGD(0.05, 0.1, 1e-6, true)
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
fmt.Println(data[0].Input, "=>", n.Predict(data[0].Input))
fmt.Println(data[5].Input, "=>", n.Predict(data[5].Input))
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

See `training/trainer_test.go` for a variety of toy examples of regression, multi-class classification, binary classification, etc.

See `examples/` for more realistic examples:

| Dataset | Topology | Epochs | Accuracy |
| ------- | -------- | ------ | -------- |
| wines   | [5 5]    | 10000  | ~98%     |
| mnist   | [50]     | 25     | ~97%     |
