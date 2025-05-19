package tokenizers

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

// 隣接トークンペアとマージ後のID保持
type Pair struct{ First, Second int }
type MergeRule struct {
	Pair Pair
	Rank int
}

// ディレクトリ内の .txt 一括読み込み
func ReadTextFilesInDir(dirPath string) (map[string]string, error) {
	result := make(map[string]string)
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.EqualFold(filepath.Ext(d.Name()), ".txt") {
			return nil
		}
		rel, err := filepath.Rel(dirPath, path) // 相対パス変換
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path) // ファイル内容取得
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", path, err)
		}
		result[rel] = string(data) // マップ登録
		return nil
	})
	return result, err
}

// BPEルール学習用の頻度統計取得
func getStats(ids []int) map[Pair]int {
	stats := make(map[Pair]int, len(ids))
	for i := 0; i+1 < len(ids); i++ {
		stats[Pair{ids[i], ids[i+1]}]++
	}
	return stats
}

// ペアをまとめて新IDに置換
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

// 学習ルールからID→バイト列辞書生成
func buildVocab(rules []MergeRule) map[int][]byte {
	vocab := make(map[int][]byte, len(rules)+256)
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}
	for _, r := range rules {
		a, ok1 := vocab[r.Pair.First]
		b, ok2 := vocab[r.Pair.Second]
		if !ok1 || !ok2 {
			continue
		}
		vocab[r.Rank] = append(append([]byte{}, a...), b...)
	}
	return vocab
}

// 学習済みルールによるエンコード処理
func Encode(text string, rules []MergeRule) []int {
	tokens := ConvertBytesToInts([]byte(text))
	for {
		if len(tokens) < 2 {
			break
		}
		stats := getStats(tokens)

		bestRank := math.MaxInt
		var bestPair Pair
		for _, r := range rules {
			if cnt, ok := stats[r.Pair]; ok && cnt > 0 && r.Rank < bestRank {
				bestRank = r.Rank
				bestPair = r.Pair
			}
		}
		if bestRank == math.MaxInt {
			break
		}

		newTokens := merge(tokens, bestPair, bestRank)
		if len(newTokens) == len(tokens) {
			break
		}
		tokens = newTokens
	}
	return tokens
}

// トークン列から文字列復元
func Decode(ids []int, rules []MergeRule) string {
	vocab := buildVocab(rules)
	var buf []byte
	for _, id := range ids {
		if seq, ok := vocab[id]; ok {
			buf = append(buf, seq...)
		}
	}
	return string(buf)
}

// バイト列→ID列変換
func ConvertBytesToInts(b []byte) []int {
	out := make([]int, len(b))
	for i, c := range b {
		out[i] = int(c)
	}
	return out
}

// テキストからBPEルール学習スライス生成
func Train(text string, maxSteps int) []MergeRule {
	parts := []string{
		// 英語向け前処理パターン
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
		ids := ConvertBytesToInts([]byte(chunk))
		for i := 0; i < maxSteps; i++ {
			stats := getStats(ids)
			if len(stats) == 0 {
				break
			}
			var best Pair
			maxCnt := 0
			for p, cnt := range stats {
				if cnt > maxCnt {
					best, maxCnt = p, cnt
				}
			}
			ids = merge(ids, best, nextID)
			rules = append(rules, MergeRule{Pair: best, Rank: nextID})
			nextID++
		}
	}

	sort.Slice(rules, func(i, j int) bool {
		return rules[i].Rank < rules[j].Rank
	})
	return rules
}
