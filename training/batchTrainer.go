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
	*batchTraining
	verbosity   int
	batchSize   int
	parallelism int
}

type batchTraining struct {
	deltas            [][][]float64
	partialDeltas     [][][][]float64
	accumulatedDeltas [][][]float64
	oldDeltas         [][][]float64
}

func newBatchTraining(layers []*deep.Layer, parallelism int) *batchTraining {
	deltas := make([][][]float64, parallelism)
	partialDeltas := make([][][][]float64, parallelism)
	accumulatedDeltas := make([][][]float64, len(layers))
	oldDeltas := make([][][]float64, len(layers))
	for w := 0; w < parallelism; w++ {
		deltas[w] = make([][]float64, len(layers))
		partialDeltas[w] = make([][][]float64, len(layers))

		for i, l := range layers {
			deltas[w][i] = make([]float64, len(l.Neurons))
			oldDeltas[i] = make([][]float64, len(l.Neurons))
			accumulatedDeltas[i] = make([][]float64, len(l.Neurons))
			partialDeltas[w][i] = make([][]float64, len(l.Neurons))
			for j, n := range l.Neurons {
				oldDeltas[i][j] = make([]float64, len(n.In))
				partialDeltas[w][i][j] = make([]float64, len(n.In))
				accumulatedDeltas[i][j] = make([]float64, len(n.In))
			}
		}
	}
	return &batchTraining{
		deltas:            deltas,
		oldDeltas:         oldDeltas,
		partialDeltas:     partialDeltas,
		accumulatedDeltas: accumulatedDeltas,
	}
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
	t.batchTraining = newBatchTraining(n.Layers, t.parallelism)

	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Examples)
	nets := make([]*deep.Neural, t.parallelism)
	currentWeights := n.Weights()

	wg := sync.WaitGroup{}
	for i := 0; i < t.parallelism; i++ {
		nets[i] = deep.NewNeural(n.Config)

		go func(id int, workCh <-chan Examples) {
			deltas := make([][]float64, len(n.Layers)) // Avoid alloc
			for i, l := range n.Layers {
				deltas[i] = make([]float64, len(l.Neurons))
			}

			for batch := range workCh {
				nets[id].ApplyWeights(currentWeights)
				for _, e := range batch {
					nets[id].Forward(e.Input)
					t.calculateDeltas(nets[id], e.Response, id)
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

		currentWeights = n.Weights()

		for i := 0; i < len(batches); i++ {
			workloads := batches[i].SplitN(t.parallelism)
			wg.Add(t.parallelism)
			for _, workload := range workloads {
				workCh <- workload
			}
			wg.Wait()

			for w := range t.partialDeltas {
				for i := range t.partialDeltas[w] {
					for j := range t.partialDeltas[w][i] {
						for k, v := range t.partialDeltas[w][i][j] {
							t.accumulatedDeltas[i][j][k] += v
							t.partialDeltas[w][i][j][k] = 0
						}
					}
				}
			}

			batchSize := float64(len(batches[i]))
			t.update(n, t.lr/batchSize, t.lr*t.lambda/float64(n.Config.Inputs), t.momentum)
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

func (t *BatchTrainer) calculateDeltas(n *deep.Neural, ideal []float64, wid int) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.deltas[wid][len(n.Layers)-1][i] = deep.Act(neuron.A).Df(neuron.Value) * (ideal[i] - neuron.Value)
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * t.deltas[wid][i+1][k]
			}
			t.deltas[wid][i][j] = deep.Act(neuron.A).Df(neuron.Value) * sum
		}
	}

	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				t.partialDeltas[wid][i][j][k] += t.deltas[wid][i][j] * l.Neurons[j].In[k].In
			}
		}
	}
}

func (t *BatchTrainer) update(n *deep.Neural, lr, lambda, momentum float64) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				delta := lr*t.accumulatedDeltas[i][j][k] - l.Neurons[j].In[k].Weight*lambda
				l.Neurons[j].In[k].Weight += delta + momentum*t.oldDeltas[i][j][k]
				t.oldDeltas[i][j][k] = delta
				t.accumulatedDeltas[i][j][k] = 0
			}
		}
	}
}
