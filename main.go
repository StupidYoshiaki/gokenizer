package main

import (
	"fmt"
	"os"
)

type Pair struct {
	First  byte
	Second byte
}

func getStats(ids []byte) map[Pair]byte {
	stats := make(map[Pair]byte)
	for i := 0; i < len(ids)-1; i++ {
		p := Pair{First: ids[i], Second: ids[i+1]}
		stats[p] = stats[p] + 1
	}
	return stats
}

func main() {
	filePath := "corpus/hatsukoi.txt"
	bytes, err := os.ReadFile(filePath)
	if err != nil {
		panic(err)
	}

	stats := getStats(bytes)
	fmt.Println(stats)

	// text := string(bytes)

	// for i := 0; i < len(text); {
	// 	r, size := utf8.DecodeRuneInString(text[i:])
	// 	b := []byte(text[i : i+size])
	// 	fmt.Printf("Bytes for %q: %v\n", r, b)
	// 	i += size
	// }
}
