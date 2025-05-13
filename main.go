package main

import (
	"fmt"
	"io/fs"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Pair は隣接するトークン ID のペアを表現します。
// BPE のマージルールでは、このペア (First, Second) がまとめられて新しい ID に置き換えられます。
type Pair struct {
	First  int
	Second int
}

// ReadTextFilesInDir は、dirPath 以下のディレクトリを再帰的に走査し、
// 拡張子が .txt のファイルをすべて読み込んで
// map[filePath] = fileContent の形で返します。
// filePath は絶対パスまたは dirPath からの相対パスになります。
func ReadTextFilesInDir(dirPath string) (map[string]string, error) {
	result := make(map[string]string)

	// filepath.WalkDir を使って再帰的にディレクトリを走査
	err := filepath.WalkDir(dirPath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// 走査中にエラーがあれば中断
			return err
		}
		// ディレクトリはスキップ
		if d.IsDir() {
			return nil
		}
		// 拡張子が .txt でなければスキップ（大文字小文字を無視）
		if !strings.EqualFold(filepath.Ext(d.Name()), ".txt") {
			return nil
		}
		// ファイルを読み込む
		data, err := ioutil.ReadFile(path)
		if err != nil {
			return fmt.Errorf("ファイル読み込み失敗 [%s]: %w", path, err)
		}
		// map に格納
		result[path] = string(data)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

// getStats は、与えられたトークン ID のスライスから
// すべての隣接ペアの出現回数をカウントして map にして返します。
// 例：ids = [A,B,B,C] → {(A,B):1, (B,B):1, (B,C):1}
func getStats(ids []int) map[Pair]int {
	stats := make(map[Pair]int)
	for i := 0; i+1 < len(ids); i++ {
		p := Pair{First: ids[i], Second: ids[i+1]}
		stats[p]++
	}
	return stats
}

// merge は、ids の中で pair.First, pair.Second が隣接して現れるすべての箇所を
// 新しいトークン ID idx にまとめて置き換えた新しいスライスを返します。
// これが BPE の「１ステップのマージ操作」に相当します。
func merge(ids []int, pair Pair, idx int) []int {
	out := make([]int, 0, len(ids))
	for i := 0; i < len(ids); {
		// ペアにマッチしたら新しい ID を追加し、元の２要素をスキップ
		if i+1 < len(ids) && ids[i] == pair.First && ids[i+1] == pair.Second {
			out = append(out, idx)
			i += 2
		} else {
			// マッチしなければそのまま現在のトークンをコピー
			out = append(out, ids[i])
			i++
		}
	}
	return out
}

// getVocabDict は、学習済みのマージルール mergeDict をもとに
// 「トークン ID → 実際のバイト列」の辞書を構築します。
// まず 0–255 は単一バイトで初期化し、その後 mergeDict の順にマージした
// バイト列を追加していきます。
func getVocabDict(mergeDict map[Pair]int) map[int][]byte {
	vocab := make(map[int][]byte, len(mergeDict)+256)

	// --- 初期化: 0–255 を単一バイトで登録 ---
	for i := 0; i < 256; i++ {
		vocab[i] = []byte{byte(i)}
	}

	// --- マージルールに従って辞書を拡張 ---
	// mergeDict に登録された順序は学習時の優先順位（rank）に相当
	for pair, idx := range mergeDict {
		seqA, okA := vocab[pair.First]
		seqB, okB := vocab[pair.Second]
		if !okA || !okB {
			// いずれかのエントリが未定義なら飛ばす
			continue
		}
		// バイト列を append で連結し、新しい ID に登録
		vocab[idx] = append(append([]byte{}, seqA...), seqB...)
	}
	return vocab
}

// decode は、トークン ID の列をバイト列に戻し、UTF-8 文字列として返します。
// 不正なバイト列は自動的に置換文字 (�) に変換されます。
func decode(ids []int, vocab map[int][]byte) string {
	var buf []byte
	for _, id := range ids {
		if seq, ok := vocab[id]; ok {
			buf = append(buf, seq...)
		}
	}
	return string(buf)
}

// convertBytesToInts は、[]byte を []int に変換します。
// トークン ID として扱うための準備関数です。
func convertBytesToInts(b []byte) []int {
	out := make([]int, len(b))
	for i, c := range b {
		out[i] = int(c)
	}
	return out
}

// encode は、テキストを UTF-8 バイト → []int に変換し、
// mergeDict に従って可能な限り BPE のマージを適用した結果を返します。
// 返り値は「最終的なトークン ID の列」です。
func encode(text string, mergeDict map[Pair]int) []int {
	// ① 生テキスト → バイト列 → ID 列に変換
	tokens := convertBytesToInts([]byte(text))

	// ② BPE マージ可能な限り繰り返す
	for {
		if len(tokens) < 2 {
			// トークン列が 1 つ以下ならマージ不要
			break
		}
		// ②-a 隣接ペアごとの出現カウント
		stats := getStats(tokens)

		// ②-b mergeDict から「最も優先順位の高いペア」を探す
		minRank := math.MaxInt
		var best Pair
		for pair := range stats {
			if rank, ok := mergeDict[pair]; ok && rank < minRank {
				minRank, best = rank, pair
			}
		}
		if minRank == math.MaxInt {
			// どのペアもルールに登録されていなければ終了
			break
		}

		// ②-c ペアをマージして tokens を更新
		tokens = merge(tokens, best, mergeDict[best])
	}
	return tokens
}

// train は、入力テキストを正規表現で前処理（英語向け）し、
// 各チャンクに対して BPE の学習（ペアの頻度に応じたマージルール生成）を行います。
// 実践投入用には日本語なら形態素解析を組み合わせてください。
func train(text string) map[Pair]int {
	// --- 前処理パターン (英語向け) ---
	parts := []string{
		// 単語本体＋('s, 're など) の一体化
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*` +
			`[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		// 異なる大文字パターン
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+` +
			`[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		// 数字 (1~3桁)
		`\p{N}{1,3}`,
		// 記号列＋改行/スラッシュ
		` ?[^\s\p{L}\p{N}]+[\r\n/]*`,
		// 空白 or 改行
		`\s+`,
	}
	re := regexp.MustCompile(strings.Join(parts, "|"))

	// テキストをチャンクに分割
	chunks := re.FindAllString(text, -1)

	mergeDict := make(map[Pair]int)
	nextID := 256 // 新しいトークン ID のカウンタ

	// 各チャンクごとにマージルールを学習
	for _, chunk := range chunks {
		// チャンクをバイト→ID 列に
		ids := convertBytesToInts([]byte(chunk))
		// 例として最大 100 ステップだけマージ
		for iter := 0; iter < 100; iter++ {
			stats := getStats(ids)
			if len(stats) == 0 {
				break
			}
			// 最頻出ペアを選択
			var best Pair
			maxCnt := 0
			for p, cnt := range stats {
				if cnt > maxCnt {
					maxCnt, best = cnt, p
				}
			}
			// 新 ID を割り当ててマージ＆ルール登録
			ids = merge(ids, best, nextID)
			mergeDict[best] = nextID
			nextID++
		}
	}

	return mergeDict
}

func main() {
	// コーパス読み込み
	dirPath := "corpus/oasis"
	corpusDict, err := ReadTextFilesInDir(dirPath)
	if err != nil {
		panic(err)
	}

	// コーパスを結合
	text := ""
	for _, content := range corpusDict {
		text += content
	}

	// 1) 学習フェーズ: マージルールを生成
	merges := train(text)
	fmt.Printf("学習したマージルール数: %d\n", len(merges))
	for pair, idx := range merges {
		if idx >= 256 && idx < 266 {
			fmt.Printf("  %d: (%d, %d)\n", idx, pair.First, pair.Second)
		}
	}
	os.Exit(0)

	// 2) テスト: 簡易サンプルの encode/decode
	testDirPath := "corpus/blur"
	testCorpusDict, err := ReadTextFilesInDir(testDirPath)
	if err != nil {
		panic(err)
	}

	sample := testCorpusDict["corpus/blur/girls_and_boys.txt"]
	encoded := encode(sample, merges)
	fmt.Printf("エンコード結果: %v\n", encoded)

	vocab := getVocabDict(merges)
	decoded := decode(encoded, vocab)
	fmt.Printf("デコード結果: %s\n", decoded)
}
