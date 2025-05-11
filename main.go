package main

import (
	"fmt"
	"os"
)

type Pair struct {
	First  int
	Second int
}

func getStats(ids []int) map[Pair]int {
	stats := make(map[Pair]int)
	for i := 0; i < len(ids)-1; i++ {
		p := Pair{First: ids[i], Second: ids[i+1]}
		stats[p] = stats[p] + 1
	}
	return stats
}

func merge(ids []int, pair Pair, idx int) []int {
	newIds := make([]int, 0, len(ids))
	for i := 0; i < len(ids); {
		if i+1 < len(ids) && ids[i] == pair.First && ids[i+1] == pair.Second {
			newIds = append(newIds, idx)
			i += 2
		} else {
			newIds = append(newIds, ids[i])
			i += 1
		}
	}
	return newIds
}

func main() {
	filePath := "corpus/hatsukoi.txt"
	data, err := os.ReadFile(filePath)
	if err != nil {
		panic(err)
	}

	ids := make([]int, len(data))
	for i, b := range data {
		ids[i] = int(b)
	}

	stats := getStats(ids)
	for pair, count := range stats {
		fmt.Printf("(%d, %d): %d\n", pair.First, pair.Second, count)
	}
}
