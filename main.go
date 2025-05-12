package main

import (
	"fmt"
	"os"
)

// Pair は隣接するトークン ID のペアを表します
type Pair struct {
	First  int
	Second int
}

// getStats は ids スライス中の隣接ペア出現回数をカウントして返します。
// Python の zip(ids, ids[1:]) 相当です。
func getStats(ids []int) map[Pair]int {
	stats := make(map[Pair]int)
	for i := 0; i < len(ids)-1; i++ {
		p := Pair{First: ids[i], Second: ids[i+1]}
		stats[p] = stats[p] + 1
	}
	return stats
}

// merge は ids の中から pair.First, pair.Second の隣接ペアを
// すべて idx という新しいトークン ID にまとめて返します。
func merge(ids []int, pair Pair, idx int) []int {
	newIds := make([]int, 0, len(ids))
	for i := 0; i < len(ids); {
		if i+1 < len(ids) && ids[i] == pair.First && ids[i+1] == pair.Second {
			newIds = append(newIds, idx)
			i += 2
		} else {
			newIds = append(newIds, ids[i])
			i++
		}
	}
	return newIds
}

// getVocabDict は、mergeDict で指定されたマージルール（ペア→新ID）をもとに
// 「ID→バイト列」を構築します。
// 0–255 は初期バイト、以降の ID はマージ済みバイト列として登録します。
func getVocabDict(mergeDict map[Pair]int) map[int][]byte {
	vocabDict := make(map[int][]byte, len(mergeDict)+256)

	// 0–255 は単一バイトとして初期化
	for i := 0; i < 256; i++ {
		vocabDict[i] = []byte{byte(i)}
	}

	// マージルールに従って辞書を拡張
	for pair, idx := range mergeDict {
		seq0, ok0 := vocabDict[pair.First]
		seq1, ok1 := vocabDict[pair.Second]
		if !ok0 || !ok1 {
			// 片方でも未定義ならスキップ
			continue
		}
		// バイト列を連結して新しいエントリに登録
		merged := append(append([]byte{}, seq0...), seq1...)
		vocabDict[idx] = merged
	}

	return vocabDict
}

func decode(ids []int, vocabDict map[int][]byte) string {
	var tokens []byte
	for _, idx := range ids {
		if seq, ok := vocabDict[idx]; ok {
			tokens = append(tokens, seq...)
		}
	}
	return string(tokens)
}

func main() {
	// コーパス読み込み
	filePath := "corpus/hatsukoi.txt"
	data, err := os.ReadFile(filePath)
	if err != nil {
		panic(err)
	}

	// []byte → []int に変換（ID 化）
	ids := make([]int, len(data))
	for i, b := range data {
		ids[i] = int(b)
	}

	// 隣接ペアの頻度計算
	stats := getStats(ids)
	fmt.Println("Pair frequencies:")
	for p, cnt := range stats {
		fmt.Printf("  (%d, %d): %d\n", p.First, p.Second, cnt)
	}

	// 例：最頻出ペアを１つ取得して merge を試す
	var best Pair
	maxCnt := 0
	for p, cnt := range stats {
		if cnt > maxCnt {
			maxCnt = cnt
			best = p
		}
	}
	fmt.Printf("\nMerging best pair (%d, %d)→ new ID %d\n\n", best.First, best.Second, 256)

	mergedIds := merge(ids, best, 256)
	fmt.Printf("Original length: %d, Merged length: %d\n", len(ids), len(mergedIds))

	// 例：getVocabDict で辞書を構築
	mergeRules := map[Pair]int{best: 256}
	vocab := getVocabDict(mergeRules)
	fmt.Printf("\nVocab entry for ID 256: %v\n", vocab[256])

	decoded := decode(mergedIds, vocab)
	fmt.Println(string([]rune(decoded)[:50]))
}
