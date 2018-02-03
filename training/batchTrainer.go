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
	*internalb
	verbosity   int
	batchSize   int
	parallelism int
	optimizer   Optimizer
}

type internalb struct {
	deltas            [][][]float64
	partialDeltas     [][][][]float64
	accumulatedDeltas [][][]float64
	moments           [][][]float64
}

func newBatchTraining(layers []*deep.Layer, parallelism int) *internalb {
	deltas := make([][][]float64, parallelism)
	partialDeltas := make([][][][]float64, parallelism)
	accumulatedDeltas := make([][][]float64, len(layers))
	moments := make([][][]float64, len(layers))
	for w := 0; w < parallelism; w++ {
		deltas[w] = make([][]float64, len(layers))
		partialDeltas[w] = make([][][]float64, len(layers))

		for i, l := range layers {
			deltas[w][i] = make([]float64, len(l.Neurons))
			moments[i] = make([][]float64, len(l.Neurons))
			accumulatedDeltas[i] = make([][]float64, len(l.Neurons))
			partialDeltas[w][i] = make([][]float64, len(l.Neurons))
			for j, n := range l.Neurons {
				moments[i][j] = make([]float64, len(n.In))
				partialDeltas[w][i][j] = make([]float64, len(n.In))
				accumulatedDeltas[i][j] = make([]float64, len(n.In))
			}
		}
	}
	return &internalb{
		deltas:            deltas,
		moments:           moments,
		partialDeltas:     partialDeltas,
		accumulatedDeltas: accumulatedDeltas,
	}
}

func NewBatchTrainer(optimizer Optimizer, verbosity, batchSize, parallelism int) *BatchTrainer {
	return &BatchTrainer{
		optimizer:   optimizer,
		verbosity:   verbosity,
		batchSize:   iparam(batchSize, 1),
		parallelism: iparam(parallelism, 1),
	}
}

func (t *BatchTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internalb = newBatchTraining(n.Layers, t.parallelism)

	train := make(Examples, len(examples))
	copy(train, examples)

	workCh := make(chan Example)
	nets := make([]*deep.Neural, t.parallelism)

	wg := sync.WaitGroup{}
	for i := 0; i < t.parallelism; i++ {
		nets[i] = deep.NewNeural(n.Config)

		go func(id int, workCh <-chan Example) {
			for e := range workCh {
				nets[id].Forward(e.Input)
				t.calculateDeltas(nets[id], e.Response, id)
				wg.Done()
			}
		}(i, workCh)
	}

	w := tabwriter.NewWriter(os.Stdout, 16, 0, 3, ' ', 0)
	fmt.Fprintf(w, "Epochs\tElapsed\tLoss (%s)\t\n---\t---\t---\t\n", n.Config.Loss)
	ts := time.Now()
	for i := 0; i < iterations; i++ {
		train.Shuffle()
		batches := train.SplitSize(t.batchSize)

		for i := 0; i < len(batches); i++ {
			currentWeights := n.Weights()
			for i := range nets {
				nets[i].ApplyWeights(currentWeights)
			}

			for _, item := range batches[i] {
				wg.Add(1)
				workCh <- item
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

			t.update(n)
		}

		if t.verbosity > 0 && i%t.verbosity == 0 && len(validation) > 0 {
			loss := CrossValidate(n, validation)
			fmt.Fprintf(w, "%d\t%s\t%.5f\t\n", i+t.verbosity, time.Since(ts).String(), loss)
			w.Flush()
		}
	}
}

func (t *BatchTrainer) calculateDeltas(n *deep.Neural, ideal []float64, wid int) {
	for i, neuron := range n.Layers[len(n.Layers)-1].Neurons {
		t.deltas[wid][len(n.Layers)-1][i] = deep.GetLoss(n.Config.Loss).Df(
			neuron.Value,
			ideal[i],
			neuron.DActivate(neuron.Value))
	}

	for i := len(n.Layers) - 2; i >= 0; i-- {
		for j, neuron := range n.Layers[i].Neurons {
			var sum float64
			for k, s := range neuron.Out {
				sum += s.Weight * t.deltas[wid][i+1][k]
			}
			t.deltas[wid][i][j] = neuron.DActivate(neuron.Value) * sum
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

func (t *BatchTrainer) update(n *deep.Neural) {
	for i, l := range n.Layers {
		for j := range l.Neurons {
			for k := range l.Neurons[j].In {
				update := t.optimizer.Update(l.Neurons[j].In[k].Weight,
					t.accumulatedDeltas[i][j][k],
					t.moments[i][j][k])
				l.Neurons[j].In[k].Weight += update
				t.moments[i][j][k] = update
				t.accumulatedDeltas[i][j][k] = 0
			}
		}
	}
}
