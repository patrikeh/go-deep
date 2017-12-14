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

	deep "github.com/patrikeh/go-deep"
)

/*
	mnist classifier
	mnist is a set of hand-written digits 0-9
	the dataset in a sane format (as used here) can be found at:
	https://pjreddie.com/projects/mnist-in-csv/
*/
func main() {
	rand.Seed(time.Now().UnixNano())

	train, err := load("./mnist_train.data")
	if err != nil {
		panic(err)
	}
	test, err := load("./mnist_test.data")
	if err != nil {
		panic(err)
	}

	for i := range train {
		for j := range train[i].Input {
			train[i].Input[j] = train[i].Input[j] / 255
		}
	}
	for i := range test {
		for j := range train[i].Input {
			train[i].Input[j] = train[i].Input[j] / 255
		}
	}
	test.Shuffle()
	train.Shuffle()

	neural := deep.NewNeural(&deep.Config{
		Inputs:     len(train[0].Input),
		Layout:     []int{80, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMulti,
		Weight:     deep.NewNormal(1, 0.01), // slight positive bias helps ReLU
		Error:      deep.MSE,
		Bias:       1,
	})

	train, val := train.Split(0.9)
	fmt.Printf("training: %d, val: %d, test: %d\n", len(train), len(val), len(test))

	neural.TrainWithCrossValidation(train, val, 20, 1, 0.001, 0.00001, 0.1)

	correct := 0
	for _, d := range test {
		est := neural.Predict(d.Input)
		if deep.ArgMax(d.Response) == deep.ArgMax(est) {
			correct++
		}
		fmt.Printf("want: %d guess: %d\n", deep.ArgMax(d.Response), deep.ArgMax(est))
	}
	fmt.Printf("accuracy: %.2f\n", float64(correct)/float64(len(test)))

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
	resEncoded := onehot(10, res)
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
	res[int(val)] = 1
	return res
}
