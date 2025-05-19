package main

import (
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Pair は隣接するトークン ID のペアを表現します。
type Pair struct {
	First, Second int
}

// MergeRule は学習済みペアとそのランクを保持します。
type MergeRule struct {
	Pair Pair
	Rank int
}

// ReadTextFilesInDir は dirPath 以下の .txt ファイルを再帰的に読み込み、
// map[相対パス] = ファイル内容 で返します。
func ReadTextFilesInDir(dirPath string) (map[string]string, error) {
	result := make(map[string]string)
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.EqualFold(filepath.Ext(d.Name()), ".txt") {
			return nil
		}
		// dirPath からの相対パスをキーに
		rel, err := filepath.Rel(dirPath, path)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("ファイル読み込み失敗 [%s]: %w", path, err)
		}
		result[rel] = string(data)
		return nil
	})
	return result, err
}

// getStats は隣接ペアの出現回数を数えます。
func getStats(ids []int) map[Pair]int {
	stats := make(map[Pair]int, len(ids))
	for i := 0; i+1 < len(ids); i++ {
		stats[Pair{ids[i], ids[i+1]}]++
	}
	return stats
}

// merge は pair にマッチする隣接トークンを idx にまとめます。
func merge(ids []int, pair Pair, idx int) []int {
	out := make([]int, 0, len(ids))
	for i := 0; i < len(ids); {
		if i+1 < len(ids) && ids[i] == pair.First && ids[i+1] == pair.Second {
			out = append(out, idx)
			i += 2
		} else {
			out = append(out, ids[i])
			i++
		}
	}
	return out
}

// getVocabDict は学習ルールをもとに ID→バイト列辞書を構築します。
func getVocabDict(rules []MergeRule) map[int][]byte {
	vocab := make(map[int][]byte, len(rules)+256)
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}
	for _, rule := range rules {
		a, okA := vocab[rule.Pair.First]
		b, okB := vocab[rule.Pair.Second]
		if !okA || !okB {
			continue
		}
		seq := append(append([]byte{}, a...), b...)
		vocab[rule.Rank] = seq
	}
	return vocab
}

// decode は ID 列を文字列に戻します。
func decode(ids []int, vocab map[int][]byte) string {
	var buf []byte
	for _, id := range ids {
		if seq, ok := vocab[id]; ok {
			buf = append(buf, seq...)
		}
	}
	return string(buf)
}

// convertBytesToInts はバイト列を ID 列に変換します。
func convertBytesToInts(b []byte) []int {
	out := make([]int, len(b))
	for i, c := range b {
		out[i] = int(c)
	}
	return out
}

// encode はマージルールをランク順に適用します。
func encode(text string, rules []MergeRule) []int {
	tokens := convertBytesToInts([]byte(text))
	for {
		if len(tokens) < 2 {
			break
		}
		stats := getStats(tokens)

		// 最もランクの低い（優先度高い）ルールを探す
		bestRank := math.MaxInt
		var bestPair Pair
		for _, rule := range rules {
			if cnt, ok := stats[rule.Pair]; ok && rule.Rank < bestRank && cnt > 0 {
				bestRank = rule.Rank
				bestPair = rule.Pair
			}
		}
		if bestRank == math.MaxInt {
			break // 適用可能なペアなし
		}

		newTokens := merge(tokens, bestPair, bestRank)
		// 変化がなければ終了 (無限ループ防止)
		if len(newTokens) == len(tokens) {
			break
		}
		tokens = newTokens
	}
	return tokens
}

// train はシンプルな BPE 学習を行い、MergeRule のスライスを返します。
func train(text string) []MergeRule {
	// 英語向け前処理パターン例
	parts := []string{
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*` +
			`[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+` +
			`[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`\p{N}{1,3}`,
		` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
		`\s+`,
	}
	re := regexp.MustCompile(strings.Join(parts, "|"))
	chunks := re.FindAllString(text, -1)

	var rules []MergeRule
	nextID := 256

	for _, chunk := range chunks {
		ids := convertBytesToInts([]byte(chunk))
		for iter := 0; iter < 100; iter++ {
			stats := getStats(ids)
			if len(stats) == 0 {
				break
			}
			// 最頻出ペアを選ぶ
			maxCnt := 0
			var best Pair
			for p, cnt := range stats {
				if cnt > maxCnt {
					maxCnt, best = cnt, p
				}
			}
			ids = merge(ids, best, nextID)
			rules = append(rules, MergeRule{Pair: best, Rank: nextID})
			nextID++
		}
	}
	// ランク順にソート（念のため）
	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Rank < rules[j].Rank
	})
	return rules
}

func main() {
	// コーパス読み込み
	corpus, err := ReadTextFilesInDir("corpus/oasis")
	if err != nil {
		panic(err)
	}
	var allText strings.Builder
	for _, txt := range corpus {
		allText.WriteString(txt)
	}

	// 学習
	rules := train(allText.String())
	fmt.Printf("学習したルール数: %d\n", len(rules))

	// テスト
	testCorpus, err := ReadTextFilesInDir("corpus/blur")
	if err != nil {
		panic(err)
	}
	sample := testCorpus["girls_and_boys.txt"]
	enc := encode(sample, rules)
	fmt.Printf("エンコード結果: %v\n", enc)

	vocab := getVocabDict(rules)
	dec := decode(enc, vocab)
	fmt.Printf("デコード結果: %s\n", dec)
}
