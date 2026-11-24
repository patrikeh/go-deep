package training

import (
	"math/rand"
	"runtime"
	"testing"

	deep "github.com/patrikeh/go-deep"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func Benchmark_xor(b *testing.B) {
	random := rand.New(rand.NewSource(0))
	n := newNeural(b, &deep.Config{
		Inputs:     2,
		Layout:     []int{32, 32, 1},
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.NewUniform(random, .25, 0),
		Bias:       true,
	})
	exs := Examples{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	}
	const minExamples = 4000
	var dupExs Examples
	for len(dupExs) < minExamples {
		dupExs = append(dupExs, exs...)
	}

	for i := 0; i < b.N; i++ {
		const iterations = 20
		solver := NewAdam(0.001, 0.9, 0.999, 1e-8)
		trainer := NewBatchTrainer(solver, 0, len(dupExs)/2, runtime.NumCPU(), random)
		require.NoError(b, trainer.Train(n, dupExs, dupExs, iterations))
	}
}

func batchTestNet(t testing.TB) *deep.Neural {
	var i float64
	weight := func() float64 {
		i++
		return 0.1 + i/50
	}
	return newNeural(t, &deep.Config{Inputs: 1, Layout: []int{3, 1}, Weight: weight})
}

func testRand() *rand.Rand {
	return rand.New(rand.NewSource(0))
}

func Test_BatchGradientIsAveraged(t *testing.T) {
	example := Example{[]float64{0.5}, []float64{1}}
	for _, size := range []int{1, 2, 32, 128} {
		online, batched := batchTestNet(t), batchTestNet(t)
		require.NoError(t, NewTrainer(NewSGD(0.1, 0, 0, false), 0, testRand()).Train(online, Examples{example}, nil, 1))

		var copies Examples
		for range size {
			copies = append(copies, example)
		}
		require.NoError(t, NewBatchTrainer(NewSGD(0.1, 0, 0, false), 0, size, 3, testRand()).Train(batched, copies, nil, 1))

		want, got := online.Weights(), batched.Weights()
		for i := range want {
			for j := range want[i] {
				assert.InDeltaSlice(t, want[i][j], got[i][j], 1e-12, "batch size %d", size)
			}
		}
	}
}

func Test_BatchTrainerStopsItsWorkers(t *testing.T) {
	const trainings, workers = 20, 8
	before := runtime.NumGoroutine()
	for range trainings {
		require.NoError(t, NewBatchTrainer(NewSGD(0.1, 0, 0, false), 0, 1, workers, testRand()).Train(batchTestNet(t), Examples{{[]float64{0}, []float64{0}}}, nil, 1))
	}
	// workers may still be returning: yield to them rather than sleep
	for i := 0; i < 100_000 && runtime.NumGoroutine() > before; i++ {
		runtime.Gosched()
	}
	assert.LessOrEqual(t, runtime.NumGoroutine(), before)
}

func Test_TrainLeavesExamplesInOrder(t *testing.T) {
	var examples Examples
	for i := range 16 {
		examples = append(examples, Example{[]float64{float64(i)}, []float64{0}})
	}
	original := append(Examples(nil), examples...)
	require.NoError(t, NewTrainer(NewSGD(0.1, 0, 0, false), 0, testRand()).Train(batchTestNet(t), examples, nil, 3))
	assert.Equal(t, original, examples)
}

func Test_TrainRejectsInvalidArguments(t *testing.T) {
	sgd := func() Solver { return NewSGD(0.1, 0, 0, false) }
	good := Examples{{[]float64{1}, []float64{1}}}
	online := func(s Solver, r *rand.Rand) Trainer { return NewTrainer(s, 1, r) }
	batch := func(s Solver, r *rand.Rand) Trainer { return NewBatchTrainer(s, 1, 1, 2, r) }

	tests := map[string]struct {
		trainer    func(Solver, *rand.Rand) Trainer
		solver     Solver
		random     *rand.Rand
		examples   Examples
		validation Examples
		iterations int
		want       error
	}{
		"online, short input":        {online, sgd(), testRand(), Examples{{[]float64{}, []float64{1}}}, nil, 1, deep.ErrDimension},
		"online, long response":      {online, sgd(), testRand(), Examples{{[]float64{1}, []float64{1, 2}}}, nil, 1, deep.ErrDimension},
		"online, invalid validation": {online, sgd(), testRand(), good, Examples{{[]float64{1, 2}, []float64{1}}}, 1, deep.ErrDimension},
		"online, no solver":          {online, nil, testRand(), good, nil, 1, ErrInvalidTraining},
		"online, no random source":   {online, sgd(), nil, good, nil, 1, ErrInvalidTraining},
		"online, negative epochs":    {online, sgd(), testRand(), good, nil, -1, ErrInvalidTraining},
		"batch, short input":         {batch, sgd(), testRand(), Examples{{[]float64{}, []float64{1}}}, nil, 1, deep.ErrDimension},
		"batch, invalid validation":  {batch, sgd(), testRand(), good, Examples{{[]float64{1}, []float64{}}}, 1, deep.ErrDimension},
		"batch, no random source":    {batch, sgd(), nil, good, nil, 1, ErrInvalidTraining},
		"batch, negative batch size": {func(s Solver, r *rand.Rand) Trainer { return NewBatchTrainer(s, 1, -1, 2, r) }, sgd(), testRand(), good, nil, 1, ErrInvalidTraining},
		"batch, negative workers":    {func(s Solver, r *rand.Rand) Trainer { return NewBatchTrainer(s, 1, 1, -1, r) }, sgd(), testRand(), good, nil, 1, ErrInvalidTraining},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			n := batchTestNet(t)
			before := n.Weights()
			err := test.trainer(test.solver, test.random).Train(n, test.examples, test.validation, test.iterations)
			assert.ErrorIs(t, err, test.want)
			assert.Equal(t, before, n.Weights(), "weights changed before failing")
		})
	}
}
