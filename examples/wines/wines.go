package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
)

func main() {

	rand.Seed(time.Now().UnixNano())

	data, err := load("./wine.data")
	if err != nil {
		panic(err)
	}

	for i := range data {
		deep.Standardize(data[i].Input)
	}
	data.Shuffle()

	fmt.Printf("have %d entries\n", len(data))

	neural := deep.NewNeural(&deep.Config{
		Inputs:     len(data[0].Input),
		Layout:     []int{8, 3},
		Activation: deep.ActivationTanh,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(1, 0),
		Bias:       true,
	})

	//trainer := training.NewBatchTrainer(training.NewSGD(0.005, 0.1, true), 50, 300, 4)
	trainer := training.NewTrainer(training.NewSGD(0.005, 0.5, true), 50)

	//data, heldout := data.Split(0.5)
	heldout := data
	trainer.Train(neural, data, data, 7500)

	correct := 0
	for _, d := range heldout {
		est := neural.Predict(d.Input)
		if deep.ArgMax(d.Response) == deep.ArgMax(est) {
			correct++
		}
	}
	fmt.Printf("accuracy: %.2f\n", float64(correct)/float64(len(heldout)))
}

func load(path string) (training.Examples, error) {
	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(bufio.NewReader(f))

	var examples training.Examples
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		examples = append(examples, toExample(record))
	}

	return examples, nil
}

func toExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[0], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(3, res)
	var features []float64
	for i := 1; i < len(in); i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)-1] = 1
	return res
}
