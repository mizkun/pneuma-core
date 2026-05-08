// Akasha configuration for pneuma-core
//
// Akasha は pneuma-core の "設計ハーネス"。Python ランタイム (src/pneuma_core/)
// が一次ソースで、Akasha 側は story / contract / config を保持する。
//
// 仕様は Akasha main (packages/akasha/src/config/types.ts, story/schema.ts,
// contracts/define.ts) の AkashaConfig / Story / ContractSpec に準拠。

import { defineConfig } from "@mizkun/akasha";

export default defineConfig({
  product: "pneuma-core",

  // Story の検索パス。story:validate のデフォルト glob。
  // Product / Domain は明示的に story:validate サブコマンドで対象を指定する。
  storyDir: "akasha/**/story.yaml",

  // Contract Pack の TS スキャン対象 (defineContract の静的検出)。
  // 主たる登録は contractEntry (.mjs) で行うが、TS 仕様も並置している。
  contractDir: "akasha/contracts/**/*.mjs",

  // Product Story 本体。dashboard / on-pr が --story-path 省略時の既定として参照。
  productStoryPath: "akasha/product/story.yaml",

  // Python リポジトリ。story-update チェックが「コード変更」と見なす拡張子。
  codeExtensions: [".py"],

  // Contract Pack のホスト側エントリ。
  // Akasha がロード時に dynamic import し、defineContract() を実行して
  // singleton registry に登録する。
  contractEntry: "./akasha/contracts/index.mjs",

  // touched code path → story.yaml のマッピング。
  // src/pneuma_core/<domain>/ 配下の変更を、対応するドメイン story に紐づける。
  storyMappings: [
    { source: "src/pneuma_core/models",    story: "akasha/domains/models/story.yaml" },
    { source: "src/pneuma_core/emotion",   story: "akasha/domains/emotion/story.yaml" },
    { source: "src/pneuma_core/memory",    story: "akasha/domains/memory/story.yaml" },
    { source: "src/pneuma_core/runtime",   story: "akasha/domains/runtime/story.yaml" },
    { source: "src/pneuma_core/llm",       story: "akasha/domains/llm/story.yaml" },
    { source: "src/pneuma_core/storage",   story: "akasha/domains/storage/story.yaml" },
    { source: "src/pneuma_core/protocols", story: "akasha/domains/protocols/story.yaml" },
  ],
});
