# go-deep

[![Go Reference](https://pkg.go.dev/badge/github.com/patrikeh/go-deep.svg)](https://pkg.go.dev/github.com/patrikeh/go-deep)
[![Go Report Card](https://goreportcard.com/badge/github.com/patrikeh/go-deep)](https://goreportcard.com/report/github.com/patrikeh/go-deep)
[![CI](https://github.com/patrikeh/go-deep/actions/workflows/ci.yml/badge.svg)](https://github.com/patrikeh/go-deep/actions/workflows/ci.yml)

Feed forward/backpropagation neural network implementation. Currently supports:

- Activation functions: sigmoid, hyperbolic, ReLU
- Solvers: SGD, SGD with momentum/nesterov, Adam
- Classification modes: regression, multi-class, multi-label, binary
- Supports batch training in parallel
- Bias nodes

Networks are modeled as a set of neurons connected through synapses. No GPU computations - don't use this for any large scale applications.

## Install

```
go get github.com/patrikeh/go-deep
```

## Usage

Import the go-deep package

```go
import (
	"fmt"
	"math/rand"

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
// All randomness comes from here, so a fixed seed makes training reproducible
random := rand.New(rand.NewSource(0))

n, err := deep.NewNeural(&deep.Config{
	/* Input dimensionality */
	Inputs: 2,
	/* Two hidden layers consisting of two neurons each, and a single output */
	Layout: []int{2, 2, 1},
	/* Activation functions: Sigmoid, Tanh, ReLU, Linear */
	Activation: deep.ActivationSigmoid,
	/* Determines output layer activation & loss function:
	ModeRegression: linear outputs with MSE loss
	ModeMultiClass: softmax output with Cross Entropy loss
	ModeMultiLabel: sigmoid output with binary CE loss
	ModeBinary: sigmoid output with binary CE loss */
	Mode: deep.ModeBinary,
	/* Weight initializers: {deep.NewNormal(random, σ, μ), deep.NewUniform(random, width, μ)} */
	Weight: deep.NewNormal(random, 1.0, 0.0),
	/* Apply bias */
	Bias: true,
})
if err != nil {
	return err
}
```

Train:

```go
// params: learning rate, momentum, alpha decay, nesterov
optimizer := training.NewSGD(0.05, 0.1, 1e-6, true)
// params: optimizer, verbosity (print stats at every 50th iteration), random source for shuffling
trainer := training.NewTrainer(optimizer, 50, random)

train, heldout := data.Split(0.5, random)
if err := trainer.Train(n, train, heldout, 1000); err != nil { // training, validation, iterations
	return err
}
```

resulting in:

```
Epochs          Elapsed         Loss (BinCE)
---             ---             ---
50              49.625µs        0.5124
100             168.333µs       0.3342
150             224.125µs       0.2097
...
1000            1.126875ms      0.0171
```

Finally, make some predictions:

```go
for _, e := range []training.Example{data[0], data[5]} {
	prediction, err := n.Predict(e.Input)
	if err != nil {
		return err
	}
	fmt.Println(e.Input, "=>", prediction)
}
```

Alternatively, batch training can be performed in parallel:

```go
optimizer := training.NewAdam(0.001, 0.9, 0.999, 1e-8)
// params: optimizer, verbosity (print info at every n:th iteration), batch-size, number of workers, random source
trainer := training.NewBatchTrainer(optimizer, 1, 200, 4, random)

train, heldout := data.Split(0.75, random)
if err := trainer.Train(n, train, heldout, 1000); err != nil { // training, validation, iterations
	return err
}
```

Save a trained network with `n.Marshal()`, and load it with `deep.Unmarshal`.

## Examples

See `training/trainer_test.go` for a variety of toy examples of regression, multi-class classification, binary classification, etc.

See `examples/` for more realistic examples:

| Dataset | Topology | Epochs | Accuracy on held-out data |
| ------- | -------- | ------ | ------------------------- |
| wines   | [8]      | 1000   | ~97%                      |
| mnist   | [50]     | 25     | ~97%                      |
