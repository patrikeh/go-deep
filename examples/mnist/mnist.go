// Command mnist classifies hand-written digits 0-9.
//
// The dataset in the CSV format used here can be found at
// https://pjreddie.com/projects/mnist-in-csv/
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
	trainPath       = "./mnist_train.data"
	testPath        = "./mnist_test.data"
	classes         = 10
	firstLabel      = 0
	maxPixel        = 255
	validationShare = 0.1
	epochs          = 25
	reportEvery     = 1
	batchSize       = 200
	parallelism     = 8
	adamLearnRate   = 0.001
	hiddenNeurons   = 50
	weightDeviation = 0.1
)

func main() {
	if err := run(time.Now().UnixNano()); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(seed int64) error {
	random := rand.New(rand.NewSource(seed))

	train, err := dataset.LoadCSV(trainPath, classes, firstLabel)
	if err != nil {
		return err
	}
	test, err := dataset.LoadCSV(testPath, classes, firstLabel)
	if err != nil {
		return err
	}
	for _, set := range []training.Examples{train, test} {
		for _, e := range set {
			for j := range e.Input {
				e.Input[j] /= maxPixel
			}
		}
	}
	// tune on a validation set, so that the test set is only used once, at the end
	validation, train := train.Split(validationShare, random)
	fmt.Printf("training: %d, validation: %d, test: %d\n", len(train), len(validation), len(test))

	neural, err := deep.NewNeural(&deep.Config{
		Inputs:     len(train[0].Input),
		Layout:     []int{hiddenNeurons, classes},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(random, weightDeviation, 0),
		Bias:       true,
	})
	if err != nil {
		return err
	}

	trainer := training.NewBatchTrainer(training.NewAdam(adamLearnRate, 0, 0, 0), reportEvery, batchSize, parallelism, random)
	if err := trainer.Train(neural, train, validation, epochs); err != nil {
		return err
	}

	accuracy, err := dataset.Accuracy(neural, test)
	if err != nil {
		return err
	}
	fmt.Printf("test accuracy: %.2f%%\n", 100*accuracy)
	return nil
}
