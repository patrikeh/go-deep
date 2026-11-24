// Command wines classifies wines by cultivar, from the UCI wine dataset.
package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	deep "github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/examples/internal/dataset"
	"github.com/patrikeh/go-deep/training"
)

const (
	dataPath        = "./wine.data"
	classes         = 3
	firstLabel      = 1
	trainShare      = 0.75
	epochs          = 1000
	reportEvery     = 100
	batchSize       = 32
	parallelism     = 4
	adamLearnRate   = 0.01
	hiddenNeurons   = 8
	weightDeviation = 1
)

func main() {
	if err := run(time.Now().UnixNano()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(seed int64) error {
	random := rand.New(rand.NewSource(seed))

	data, err := dataset.LoadCSV(dataPath, classes, firstLabel)
	if err != nil {
		return err
	}
	data.Shuffle(random)
	train, test := data.Split(trainShare, random)
	dataset.Standardize(train, test)
	fmt.Printf("training on %d wines, testing on %d\n", len(train), len(test))

	neural, err := deep.NewNeural(&deep.Config{
		Inputs:     len(train[0].Input),
		Layout:     []int{hiddenNeurons, classes},
		Activation: deep.ActivationTanh,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(random, weightDeviation, 0),
		Bias:       true,
	})
	if err != nil {
		return err
	}

	trainer := training.NewBatchTrainer(training.NewAdam(adamLearnRate, 0, 0, 0), reportEvery, batchSize, parallelism, random)
	if err := trainer.Train(neural, train, train, epochs); err != nil {
		return err
	}

	accuracy, err := dataset.Accuracy(neural, test)
	if err != nil {
		return err
	}
	fmt.Printf("accuracy on unseen wines: %.1f%%\n", 100*accuracy)
	return nil
}
