# Pneuma Core

AIキャラクターに「内面」を与えるPythonフレームワーク。

性格、感情、記憶を持つキャラクターを作り、会話を重ねるほど関係性が育っていく仕組みを提供する。

## Features

- **感情が動的に変化する** -- PAD 3次元モデル + Big Five性格特性に基づき、会話内容に応じて感情がリアルタイムで変化。性格によって反応が異なり、時間とともにベースラインへ自然減衰する
- **記憶を性格に合わせて思い出す** -- エピソード記憶とセマンティック記憶をBig Fiveバイアス付きRAGで検索。共感性が高いキャラは感情に関する記憶を、知的好奇心が高いキャラは事実に関する記憶を優先的に思い出す
- **毎ターン、キャラの内面がプロンプトに自動反映** -- 性格、現在の感情、記憶、関係性を毎回のプロンプトに自動組み込み。開発者がプロンプトエンジニアリングを頑張る必要はない

## Installation

```bash
pip install pneuma-core
```

LLMアダプター込み:

```bash
pip install pneuma-core[all]
```

## Quick Start

### 1. キャラクターを定義する (YAML)

```yaml
# aine.character.yaml
name: アイネ
personality:
  openness: 0.9
  conscientiousness: 0.5
  extraversion: 0.3
  agreeableness: 0.8
  neuroticism: 0.6
profile: |
  内向的だけど好奇心が強い。
speaking_style: |
  丁寧だけど時々素が出る。
```

### 2. キャラクターと会話する

```python
import asyncio
from pneuma_core.character_sheet import CharacterSheet
from pneuma_core.runtime.engine import RuntimeEngine
from pneuma_core.llm.claude import ClaudeAdapter
from pneuma_core.storage.sqlite import SQLiteStorageBackend

async def main():
    character = CharacterSheet.load("aine.character.yaml")
    engine = RuntimeEngine(
        character=character.to_character(),
        llm=ClaudeAdapter(api_key="your-api-key"),
        storage=SQLiteStorageBackend("aine.db"),
    )
    response = await engine.chat("最近読んだ本でおすすめある？")
    print(response.content)

asyncio.run(main())
```

## Middleware

コアはシンプルに保ちつつ、ミドルウェアで拡張する。

```python
from pneuma_core.protocols.middleware import Middleware, PipelineContext
from pneuma_core.models.message import MessageInput, MessageOutput

class LoggingMiddleware:
    async def pre_process(self, message: MessageInput, context: PipelineContext) -> None:
        print(f"Received: {message.content}")

    async def post_process(
        self, message: MessageInput, output: MessageOutput, context: PipelineContext
    ) -> None:
        print(f"Response: {output.content}")

engine = RuntimeEngine(
    character=character,
    llm=llm,
    storage=storage,
    middlewares=[LoggingMiddleware()],
)
```

## Architecture

Pneuma Coreは2つのフェーズで動作する:

**Per-turn (毎ターン)**: ユーザーのメッセージを受け取るたびに、感情推定、記憶検索、コンテキスト組み立て、LLM呼び出し、状態更新を実行する。ミドルウェアパイプラインにより、各ステップの前後にカスタム処理を挿入できる。

**Per-session (セッション終了時)**: 会話終了時に、エピソード記憶の統合、セマンティック記憶の抽出、関係性の更新、日記の生成を行う。

```
Layer 0 (models)   : データモデル + ストレージプロトコル
Layer 1 (runtime)  : LLM連携 + 感情エンジン + 記憶検索 + ミドルウェア
```

## License

MIT
