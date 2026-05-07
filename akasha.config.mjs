// Akasha configuration for pneuma-core
//
// pneuma-core は Python のコアライブラリ (AIキャラクターに性格・感情・記憶・関係性を与える)。
// Akasha はここでは「ハーネス（仕様駆動・Story/Contract 駆動の設計レイヤー）」として動作する。
// Python ランタイム自体は Akasha に依存しない。

export default {
  product: {
    name: "pneuma-core",
    slug: "pneuma-core",
    storyDir: "akasha/product",
  },

  domains: {
    root: "akasha/domains",
    list: [
      "models",
      "emotion",
      "memory",
      "runtime",
      "llm",
      "storage",
      "protocols",
    ],
  },

  contracts: {
    root: "akasha/contracts",
    // Contract 群は TypeScript の型定義として記述する。
    // Python 実装に対する参照・整合チェック用の "外側の仕様"。
    // Python 側では `src/pneuma_core/protocols/` が一次ソース。
    primarySource: "src/pneuma_core/protocols",
  },

  // Pneuma Core は Python ライブラリ。Akasha はハーネスのみ。
  runtime: {
    language: "python",
    package: "pneuma_core",
    src: "src/pneuma_core",
    tests: "tests",
    testCommand: ".venv/bin/python -m pytest tests/ -x --tb=short",
  },

  // VibeFlow 由来の hard enforcement（role-based file ACL, Bash gating など）は
  // Akasha では使用しない。Akasha は story / contract / spec を基準にした
  // 設計レビューを優先し、shell や file 書き込みは妨げない。
  enforcement: {
    fileAccessControl: "off",
    shellGating: "off",
  },
};
