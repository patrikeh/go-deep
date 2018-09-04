package main

import (
	"bufio"
	"fmt"
	"math/rand"
	"os"
	"time"
	"strings"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"math"
	"io/ioutil"
)

func main() {

	rand.Seed(time.Now().UnixNano())

	data, err := load("./tic-tac-toe.data.txt")
	if err != nil {
		panic(err)
	}

	data.Shuffle()

	fmt.Printf("have %d entries\n", len(data))

	neural := deep.NewNeural(&deep.Config{
		Inputs:     9,
		Layout:     []int{18, 9, 1},
		Activation: deep.ActivationSigmoid,
		Weight:     deep.NewNormal(1, 0),
	})

	trainer := training.NewBatchTrainer(training.NewAdam(0.1, 0, 0, 0), 50, 10, 16)
	trainer.Train(neural, data, data, 1000)
	// c := 0
	// for i:=0 ; i<len(data); i++ {
	// 	v := neural.Predict(data[i].Input)
	// 	if !InLimit(v[0], data[i].Response[0], 0.4) {
	// 		fmt.Println(v, data[i].Response[0], "wrong")
	// 		c++
	// 	}
	// }
	// fmt.Println("count: ", c)
	dump, err := neural.Marshal()
	ioutil.WriteFile("n.nimi", dump, 0644)
}

func InLimit(a, b, l float64) bool {
    return math.Abs(a - b) <= l
}

func load(path string) (training.Examples, error) {
	file, err := os.Open(path)
	if err != nil {
	    panic(err)
	}
	defer file.Close()

	var examples training.Examples
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
	    examples = append(examples, toExample(scanner.Text()))
	}

	if err := scanner.Err(); err != nil {
	    panic(err)
	}
	return examples, nil
}

func toExample(in string) training.Example {

	var board []float64
	var res []float64

	for _,s := range strings.Split(in, ",") {
		if s == "o" {
			board =  append(board, 0.0)
		} else if s == "x" {
			board =  append(board, 0.5)
		} else if s == "b" {
			board =  append(board, 1.0)
		} else if s== "negative" {
			res =  append(res, 0.0)
		} else if s == "positive" {
			res =  append(res, 1.0)
		}
	}
	return training.Example{
		Response: res,
		Input:    board,
	}
}
