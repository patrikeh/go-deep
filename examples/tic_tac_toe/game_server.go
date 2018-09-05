package main

import (
    "io/ioutil"
    "log"
    "net/http"
    "html/template"
    "encoding/json"
    "github.com/patrikeh/go-deep"
    "strings"
    "strconv"
)

var templates = template.Must(template.ParseFiles("static/game.html"))
var ttt_NN *deep.Neural

type Board map[string]string

func renderTemplate(w http.ResponseWriter, tmpl string, par map[string]interface{}) {
    err := templates.ExecuteTemplate(w, tmpl+".html", par)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
    }
}

func playHandler(w http.ResponseWriter, r *http.Request) {
    board, ok := r.URL.Query()["board"]

    if !ok || len(board[0]) < 1 {
        log.Println("Url Param 'board' is missing")
        return
    }
    var m Board
    if len(board[0]) != 17 {
        m = Board{
            "b1": " ",
            "b2": " ",
            "b3": " ",
            "b4": " ",
            "b5": " ",
            "b6": " ",
            "b7": " ",
            "b8": " ",
            "b9": " ",
            "game over": "false",
        }
    } else {
        m = move(board[0])
    }
    jData, err := json.Marshal(m)
    if err != nil {
        panic(err)
    }
    w.Header().Set("Content-Type", "application/json")
    w.Write(jData)
}

func move(board string) Board {
    var rez []float64

    for _,s := range strings.Split(board, ",") {
        if s == "o" {
            rez =  append(rez, 0.5)
        } else if s == "x" {
            rez =  append(rez, 1.0)
        } else if s == " " {
            rez =  append(rez, 0.0)
        }
    }

    index := 0
    v := ttt_NN.Predict(rez)

    lrez := -1.0
    for indext, j:= range(v) {
        if j> lrez {
            lrez = j
            index = indext
        }
    }
    if rez[index] == 0.0 {
        rez[index] = 0.5
    }

    varmap := Board{
        "b1": " ",
        "b2": " ",
        "b3": " ",
        "b4": " ",
        "b5": " ",
        "b6": " ",
        "b7": " ",
        "b8": " ",
        "b9": " ",
        "game over": "false",
    }
    wb := ""
    for i, x := range(rez) {
        if x== 0.5 {
            varmap["b" + strconv.Itoa(i+1)] = "o"
            wb += "o,"
        } else if (x==1.0) {
            varmap["b" + strconv.Itoa(i+1)] = "x"
            wb += "x,"
        } else {
            wb += " ,"
        }
    }

    if varmap["b1"]+varmap["b2"]+varmap["b3"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b4"]+varmap["b5"]+varmap["b6"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b7"]+varmap["b8"]+varmap["b9"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b1"]+varmap["b4"]+varmap["b7"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b2"]+varmap["b5"]+varmap["b8"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b3"]+varmap["b6"]+varmap["b9"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b1"]+varmap["b5"]+varmap["b9"] == "ooo" {
        varmap["game over"] =  "true"
    }
    if varmap["b7"]+varmap["b5"]+varmap["b3"] == "ooo" {
        varmap["game over"] =  "true"
    }
    return varmap
}

func rootHandler(w http.ResponseWriter, r *http.Request) {

    varmap := map[string]interface{}{
        "b1": " ",
        "b2": " ",
        "b3": " ",
        "b4": " ",
        "b5": " ",
        "b6": " ",
        "b7": " ",
        "b8": " ",
        "b9": " ",
        "game over": "false",
    }
    renderTemplate(w, "game", varmap)
}

func main() {
    b, err := ioutil.ReadFile("trained/ttt_complete_for_o.nimi")
    if err != nil {
        panic(err)
    }
    ttt_NN, err = deep.Unmarshal(b)

    if err != nil {
        panic(err)
    }
    fs := http.FileServer(http.Dir("static/js"))
    fs2 := http.FileServer(http.Dir("static/css"))
    http.Handle("/js/", http.StripPrefix("/js/", fs))
    http.Handle("/css/", http.StripPrefix("/css/", fs2))
    http.HandleFunc("/play/", playHandler)
    http.HandleFunc("/", rootHandler)
    log.Fatal(http.ListenAndServe(":9090", nil))
}
