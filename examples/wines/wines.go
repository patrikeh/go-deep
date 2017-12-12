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
)

func main() {
	rand.Seed(time.Now().UnixNano())

	data, err := load("./wine.data")
	if err != nil {
		panic(err)
	}

	for i := range data {
		deep.Normalize(data[i].Input)
	}
	data.Shuffle()

	fmt.Printf("have %d entries\n", len(data))

	neural := deep.NewNeural(&deep.Config{
		Inputs:     len(data[0].Input),
		Layout:     []int{4, 3},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMulti,
		Weight:     deep.NewUniform(0.5, 0),
		Bias:       1,
		Error:      deep.MSE,
	})

	train, val := data.Split(0.65)
	neural.TrainWithCrossValidation(train, val, 10000, 50, 0.01, 0.001)

	correct := 0
	for _, d := range data {
		est := neural.Forward(d.Input)
		if deep.ArgMax(d.Response) == deep.ArgMax(est) {
			correct++
		}
		fmt.Printf("want: %d guess: %d\n", deep.ArgMax(d.Response), deep.ArgMax(est))
	}
	fmt.Printf("accuracy: %.2f\n", float64(correct)/float64(len(data)))
}

func load(path string) (deep.Examples, error) {
	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(bufio.NewReader(f))

	var examples deep.Examples
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		examples = append(examples, toExample(record))
	}

	return examples, nil
}

func toExample(in []string) deep.Example {
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

	return deep.Example{
		Response: resEncoded,
		Input:    features,
	}
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)-1] = 1
	return res
}
