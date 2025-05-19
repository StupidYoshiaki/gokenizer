// main.go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/StupidYoshiaki/gokenizer/tokenizers"
)

func main() {
	mode := flag.String("mode", "", "train or encode 指定")
	trainDir := flag.String("train-dir", "", "学習用txtディレクトリ")
	outRules := flag.String("out-rules", "merges.json", "出力マージルールJSON")
	rulesFile := flag.String("rules-file", "", "入力マージルールJSON")
	inputFile := flag.String("input-file", "", "エンコード対象テキスト")
	flag.Parse()

	switch *mode {
	case "train":
		if *trainDir == "" {
			log.Fatal("train-dir 必須")
		}
		corpus, err := tokenizers.ReadTextFilesInDir(*trainDir) // コーパス読み込み
		if err != nil {
			log.Fatalf("読み込み失敗: %v", err)
		}
		var all string
		for _, txt := range corpus {
			all += txt
		}
		rules := tokenizers.Train(all, 100) // ルール学習
		saveJSON(*outRules, rules)          // ルール保存
		fmt.Printf("保存完了: %s (%d ルール)\n", *outRules, len(rules))

	case "encode":
		if *rulesFile == "" || *inputFile == "" {
			log.Fatal("rules-file と input-file 両方必須")
		}
		rules := loadRules(*rulesFile)       // ルール読込
		data, err := os.ReadFile(*inputFile) // 入力読み込み
		if err != nil {
			log.Fatalf("読み込み失敗: %v", err)
		}
		ids := tokenizers.Encode(string(data), rules) // エンコード実行
		fmt.Println(ids)

	default:
		log.Fatal("mode は train or encode で指定")
	}
}

// JSON形式でマージルール保存
func saveJSON(path string, v interface{}) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("ファイル作成失敗: %v", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		log.Fatalf("JSONエンコード失敗: %v", err)
	}
}

// JSON形式でマージルール読み込み
func loadRules(path string) []tokenizers.MergeRule {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("ファイルオープン失敗: %v", err)
	}
	defer f.Close()
	var rules []tokenizers.MergeRule
	if err := json.NewDecoder(f).Decode(&rules); err != nil {
		log.Fatalf("JSONデコード失敗: %v", err)
	}
	return rules
}
