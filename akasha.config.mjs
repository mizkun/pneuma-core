// Akasha configuration for pneuma-core
//
// Akasha は pneuma-core の "設計ハーネス" として動作する。
// Python のランタイムは src/pneuma_core/ が一次ソースで、Akasha 側は
// Story (akasha/product/, akasha/domains/) と Contract (akasha/contracts/) を保持する。

import { defineConfig } from "@mizkun/akasha";

export default defineConfig({
  product: "pneuma-core",

  // storyDir は単一ディレクトリ。Akasha は再帰的に *.yaml / *.yml を拾う。
  // → akasha/product/story.yaml と akasha/domains/<name>/story.yaml の両方を読む。
  storyDir: "akasha",

  // contractDir は再帰スキャン対象。defineContract() の静的解析にも使う。
  contractDir: "akasha/contracts",

  // ─── 以下はプロジェクト固有の "意図" ───
  // 現時点の AkashaConfigSchema には未定義のため defineConfig() で strip されるが、
  // 将来のスキーマ拡張に備えて記録しておく (forward-compat hint)。
  productStoryPath: "akasha/product/story.yaml",
  codeExtensions: [".py"],

  // VibeFlow 由来の hard enforcement (role-based file ACL, Bash gating など) は
  // Akasha では扱わない。Akasha は story / contract を基準にした設計レビューに集中する。
});
