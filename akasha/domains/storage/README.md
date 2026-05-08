# Domain — storage

Python source: [`src/pneuma_core/storage/`](../../../src/pneuma_core/storage)

## 役割

キャラクターの内部状態 (Character / Memory / EmotionalState / Goals / Relation /
ChangeRecord / TodoItem) を永続化する層。StorageBackend Protocol で抽象化されており、
実装は差し替え可能。

## モジュール

| モジュール | 役割 |
|-----------|------|
| `backend` | `StorageBackend` Protocol の再エクスポート (一次ソースは `protocols.storage`) |
| `sqlite` | SQLite 実装 (同梱・外部依存なし) |
| `in_memory` | プロセス内 dict 実装 (テスト・実験用) |

## 不変条件

- StorageBackend を差し替えても、ランタイム挙動は同じであるべき (Protocol 規約)。
- SQLite は単一プロセス書き込み制約があるため、複数プロセスから書く構成では
  別バックエンド (例: PostgreSQL) を選ぶ必要がある。
- 個人データはローカルストレージに保存され、外部に送られない。

## 関連ドメイン

- [`models`](../models/) — 永続化対象のデータ型を提供
- [`memory`](../memory/) / [`runtime`](../runtime/) — 主要な消費者
- [`protocols`](../protocols/) — Protocol 定義の一次ソース
