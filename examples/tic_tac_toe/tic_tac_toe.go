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
	"strconv"
)

func compareexample(e1, e2 []float64 ) bool {
	for n1, _:= range(e1) {
		if (e1[n1] != e2[n1]) {
			return false;
		}
	}
	return true
}

func main() {

	rand.Seed(time.Now().UnixNano())

	data, err := load("./tictac_single.txt")
	if err != nil {
		panic(err)
	}

	// for n1, j:= range(data) {
	// 	for n2, x:= range(data) {
	// 		if n1 != n2 && compareexample(j.Input,x.Input) {
	// 			fmt.Println(x)
	// 		}
	// 	}
	// }
	//data.Shuffle()
	//fmt.Println(data)

	fmt.Printf("have %d entries\n", len(data))

	neural := deep.NewNeural(&deep.Config{
		Inputs:     9,
		Layout:     []int{80, 27, 9},
		Activation: deep.ActivationSigmoid,
		Mode:       deep.ModeBinary,
		Weight:     deep.NewNormal(1, 0),
		//Bias:       true,
	})

	trainer := training.NewBatchTrainer(training.NewAdam(0.02, 0.9, 0.999, 1e-8), 10, 500, 16)
	trainer.Train(neural, data, data, 6000)
	// c := 0
	// for i:=0 ; i<len(data); i++ {
	// 	v := neural.Predict(data[i].Input)
	// 	output := 0
	// 	for k:=0 ; k < 9 ; k++ {
	// 		if data[i].Response[k] == 1 {
	// 			output = k
	// 		}
	// 	}
	// 	rez := 0.0
	// 	index := 0
	// 	for indext, j:= range(v) {
	// 		if j> rez {
	// 			rez = j
	// 			index = indext
	// 		}
	// 	}

	// 	if index != output {
	// 		 c++
	// 		 //fmt.Println(v, "output:", output, "index: ", index)
	// 		 //fmt.Println(data[i].Response)
	// 	}

	// }
	//fmt.Println("count: ", c)
	dump, err := neural.Marshal()
	ioutil.WriteFile("t3.nimi", dump, 0644)
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
	var parse []float64
	//mt.Println(in)
	for _,s := range strings.Split(in, " ") {
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			panic(err)
		}
		intv := (int)(v)
		if intv == -1 {
			board =  append(board, 0.5)
		} else if intv == 1 {
			board =  append(board, 1.0)
		} else if intv == 0 {
			board =  append(board, 0.0)
		} else {
			board =  append(board, 0.0)
		}
		parse = append(parse, v)

	}

	y := (int)(parse[len(parse)-1])
	board = board[:len(board)-1]

	if len(board) != 9 {
		fmt.Println(board, res)
		panic("wrong")
	}

	for h:=0 ; h<9 ; h++ {
		if y==h {
			res = append(res, 1.0)
		} else {
			res = append(res, 0.0)
		}
	}
	return training.Example{
		Response: res,
		Input:    board,
	}
}
