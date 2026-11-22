package training

import (
	"slices"
	"sync"
	"time"

	deep "github.com/patrikeh/go-deep"
)

// BatchTrainer implements parallelized batch training
type BatchTrainer struct {
	*internalb
	verbosity   int
	batchSize   int
	parallelism int
	solver      Solver
	printer     *StatsPrinter
}

type internalb struct {
	deltas            [][][]float64
	partialDeltas     [][][][]float64
	accumulatedDeltas [][][]float64
}

func newBatchTraining(layers []*deep.Layer, parallelism int) *internalb {
	deltas := make([][][]float64, parallelism)
	partialDeltas := make([][][][]float64, parallelism)
	accumulatedDeltas := make([][][]float64, len(layers))
	for w := 0; w < parallelism; w++ {
		deltas[w] = make([][]float64, len(layers))
		partialDeltas[w] = make([][][]float64, len(layers))

		for i, l := range layers {
			deltas[w][i] = make([]float64, len(l.Neurons))
			accumulatedDeltas[i] = make([][]float64, len(l.Neurons))
			partialDeltas[w][i] = make([][]float64, len(l.Neurons))
			for j, n := range l.Neurons {
				partialDeltas[w][i][j] = make([]float64, len(n.In))
				accumulatedDeltas[i][j] = make([]float64, len(n.In))
			}
		}
	}
	return &internalb{
		deltas:            deltas,
		partialDeltas:     partialDeltas,
		accumulatedDeltas: accumulatedDeltas,
	}
}

// NewBatchTrainer returns a BatchTrainer
func NewBatchTrainer(solver Solver, verbosity, batchSize, parallelism int) *BatchTrainer {
	return &BatchTrainer{
		solver:      solver,
		verbosity:   verbosity,
		batchSize:   iparam(batchSize, 1),
		parallelism: iparam(parallelism, 1),
		printer:     NewStatsPrinter(),
	}
}

// Train trains n
func (t *BatchTrainer) Train(n *deep.Neural, examples, validation Examples, iterations int) {
	t.internalb = newBatchTraining(n.Layers, t.parallelism)

	train := slices.Clone(examples)

	nets := make([]*deep.Neural, t.parallelism)
	for i := range nets {
		nets[i] = deep.NewNeural(n.Config)
	}

	t.printer.Init(n)
	t.solver.Init(n.NumWeights())

	ts := time.Now()
	for it := 1; it <= iterations; it++ {
		train.Shuffle()
		batches := train.SplitSize(t.batchSize)

		for _, b := range batches {
			currentWeights := n.Weights()
			for _, n := range nets {
				n.ApplyWeights(currentWeights)
			}

			t.calculateBatch(nets, b)

			for _, wPD := range t.partialDeltas {
				for i, iPD := range wPD {
					iAD := t.accumulatedDeltas[i]
					for j, jPD := range iPD {
						jAD := iAD[j]
						for k, v := range jPD {
							jAD[k] += v
							jPD[k] = 0
						}
					}
				}
			}

			t.update(n, it, len(b))
		}

		if t.verbosity > 0 && it%t.verbosity == 0 && len(validation) > 0 {
			t.printer.PrintProgress(n, validation, time.Since(ts), it)
		}
	}
}

// calculateBatch deals the batch out round robin, so that the result doesn't depend on scheduling.
func (t *BatchTrainer) calculateBatch(nets []*deep.Neural, batch Examples) {
	workers := min(len(nets), len(batch))
	var wg sync.WaitGroup
	for id := range workers {
		wg.Go(func() {
			for i := id; i < len(batch); i += workers {
				nets[id].Forward(batch[i].Input)
				t.calculateDeltas(nets[id], batch[i].Response, id)
			}
		})
	}
	wg.Wait()
}

func (t *BatchTrainer) calculateDeltas(n *deep.Neural, ideal []float64, wid int) {
	deltas := t.deltas[wid]
	partialDeltas := t.partialDeltas[wid]
	last := len(n.Layers) - 1
	outputDeltas(n.Layers[last], n.Config.Loss, ideal, deltas[last])

	for i := len(n.Layers) - 2; i >= 0; i-- {
		l := n.Layers[i]
		iD := deltas[i]
		nextD := deltas[i+1]
		for j, n := range l.Neurons {
			var sum float64
			for k, s := range n.Out {
				sum += s.Weight * nextD[k]
			}
			iD[j] = n.DActivate(n.Value) * sum
		}
	}

	for i, l := range n.Layers {
		iD := deltas[i]
		iPD := partialDeltas[i]
		for j, n := range l.Neurons {
			jD := iD[j]
			jPD := iPD[j]
			for k, s := range n.In {
				jPD[k] += jD * s.In
			}
		}
	}
}

func (t *BatchTrainer) update(n *deep.Neural, it, batchSize int) {
	var idx int
	for i, l := range n.Layers {
		iAD := t.accumulatedDeltas[i]
		for j, n := range l.Neurons {
			jAD := iAD[j]
			for k, s := range n.In {
				update := t.solver.Update(s.Weight,
					jAD[k]/float64(batchSize),
					it,
					idx)
				s.Weight += update
				jAD[k] = 0
				idx++
			}
		}
	}
}
