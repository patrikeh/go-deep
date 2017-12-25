package training

import (
	"fmt"
	"os"
	"sync"
	"text/tabwriter"
	"time"

	deep "github.com/patrikeh/go-deep"
)

type BatchTrainer struct {
	lr, lambda, momentum float64
	*training
	verbosity   int
	batchSize   int
	parallelism int
}

func NewBatchTrainer(lr, lambda, momentum float64, verbosity, batchSize, parallelism int) *BatchTrainer {
	return &BatchTrainer{
		lr:          lr,
		lambda:      lambda,
		momentum:    momentum,
		verbosity:   verbosity,
		batchSize:   batchSize,
		parallelism: parallelism,
	}
}

func (t *BatchTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.training = newTraining(n.Layers)

	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Examples)
	nets := make([]*deep.Neural, t.parallelism)

	wg := sync.WaitGroup{}
	for i := 0; i < t.parallelism; i++ {
		nets[i] = deep.NewNeural(n.Config)
		go func(id int, workCh <-chan Examples) {
			for batch := range workCh {
				nets[id].ApplyWeights(n.Weights())
				for _, e := range batch {
					nets[id].Forward(e.Input)
					t.CalculateDeltas(nets[id], e.Response)
				}
				wg.Done()
			}
		}(i, workCh)
	}

	w := tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)
	fmt.Fprint(w, "Epochs\tElapsed\tError\t\n---\t---\t---\t\n")
	ts := time.Now()
	for i := 0; i < iterations; i++ {
		train.Shuffle()
		batches := examples.SplitSize(t.batchSize)
		for i := 0; i < len(batches); i++ {
			workloads := batches[i].SplitN(t.parallelism)
			wg.Add(t.parallelism)
			for _, workload := range workloads {
				workCh <- workload
			}

			wg.Wait()

			t.Update(n, t.lr, t.lambda/float64(n.Config.Inputs), t.momentum, float64(len(batches[i])))
		}

		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			rms := t.CrossValidate(n, validation)
			fmt.Fprintf(w, "%d\t%s\t%.5f\t\n", i+t.verbosity, time.Since(ts).String(), rms)
			w.Flush()
		}
	}
}

func (t *BatchTrainer) CrossValidate(n *deep.Neural, validation Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}
	return n.Config.Error(responses, predictions)
}

func (t *BatchTrainer) CalculateDeltas(n *deep.Neural, ideal []float64) {
	deltas := make([][]float64, len(n.Layers))
	for i, l := range n.Layers {
		deltas[i] = make([]float64, len(l.Neurons))
	}

	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		deltas[len(n.Layers)-1][i] = deep.Act(neuron.A).Df(neuron.Value) * (ideal[i] - neuron.Value)
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * deltas[i+1][k]
			}

			deltas[i][j] = deep.Act(neuron.A).Df(neuron.Value) * sum
		}
	}

	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				t.accumulatedDeltas[i][j][k] += deltas[i][j] * l.Neurons[j].In[k].In
			}
		}
	}
	return
}

func (t *BatchTrainer) Update(n *deep.Neural, lr, lambda, momentum, batchSize float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := (1/batchSize)*lr*t.accumulatedDeltas[i][j][k] - l.Neurons[j].In[k].Weight*lr*lambda
				l.Neurons[j].In[k].Weight += delta + momentum*t.oldDeltas[i][j][k]
				t.oldDeltas[i][j][k] = delta
				t.accumulatedDeltas[i][j][k] = 0
			}
		}
	}
}
