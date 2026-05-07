// Akasha Contract — EmbeddingService
//
// Python 一次ソース: src/pneuma_core/protocols/embedding.py
// テキスト → ベクトル変換の境界。記憶検索 (memory ドメイン) が依存する。

export interface EmbeddingService {
  embed(text: string): Promise<number[]>;
  embed_batch(texts: string[]): Promise<number[][]>;
}
