"""MockLLMAdapter: ANTHROPIC_API_KEY 未設定環境で動かすための疑似 LLM.

System prompt の特徴語からリクエスト種別を判定して、
それっぽい JSON 応答を返す:

- "感情分析" / "pleasure"+"arousal" → PAD JSON
- "記憶管理" / "episodic_memories" → SessionEndPipeline JSON
- それ以外 → speech/thought/action JSON（性格に応じた応答プールから抽出）

性格パラメータが取れるよう system_prompt にキャラ名・extraversion 等を埋め込んで
おくと、それに応じた応答プールから選ぶ。
"""

from __future__ import annotations

import asyncio
import json
import random
import re

from pneuma_core.llm.adapter import LLMRequest, LLMResponse

# キャラ名 → 応答プール
_RESPONSE_POOL: dict[str, list[tuple[str, str]]] = {
    "なでしこ": [
        ("わー、すごーい！それ気になる！", "目を輝かせている"),
        ("えー、ほんとに!?やってみたいなぁ", "両手を握りしめる"),
        ("わたしも一緒に行きたーい！", "笑顔で前のめりになる"),
        ("お肉、いいねぇ……お腹空いてきちゃった", "お腹をさする"),
        ("ねぇねぇ、今度キャンプ行こうよ！", "弾むように"),
        ("お、ちかちゃん、それヤバくない？", "目を丸くして"),
        ("えへへ、わたしそういうの好きだなぁ", "照れ笑いで頬をかく"),
        ("わー、いい匂いがしてきた気がする！", "鼻をくんくんさせる"),
        ("ふわぁ……ねむくなってきちゃった", "目をこする"),
        ("あおいちゃん優しいー、ありがと", "嬉しそうに頷く"),
        ("え、それなにそれ初めて聞いた！", "前のめりに"),
        ("リンちゃん天才かも〜！", "ぱちぱち手を叩く"),
        ("んー、よくわかんないけど楽しそう！", "首を傾げる"),
        ("わたし、ぜったい一緒にやる！", "腕まくり"),
    ],
    "千明": [
        ("おっ、それ次の活動で使えるんじゃない？", "ニヤリ"),
        ("ふっふっふ、いいこと思いついた！", "腕組み"),
        ("なでしこ、相変わらず食い意地はってんなー", "肘でつつく"),
        ("計画たてようぜ、計画！", "ノートを取り出す"),
        ("これは……アタシの出番か？", "立ち上がる"),
        ("意外と冷静だなぁ、あおい", "腕組みで頷く"),
        ("はいはい、それ却下〜", "手を振る"),
        ("おい、聞いてんのか？", "ジト目"),
        ("よし、今日はここまでにしよう", "ノートを閉じる"),
        ("なんかワクワクしてきたな", "前のめり"),
        ("ま、悪くない選択肢かもね", "腕組みで頷く"),
        ("そういうの嫌いじゃないぜ", "ニヤリと笑う"),
        ("ふーん、で？どうすんの？", "目を細める"),
        ("ハハッ、面白いじゃん！", "声を出して笑う"),
    ],
    "あおい": [
        ("ええんちゃう？ええと思うで〜", "ゆっくり頷く"),
        ("ふふ、なでしこちゃんは今日も元気やね", "微笑む"),
        ("ちかちゃん、それ無理あらへん？", "苦笑い"),
        ("お、それええアイデアやんか", "ぱっと顔を上げる"),
        ("そやなぁ、ちょっと考えとこか", "頬に手を当てる"),
        ("ふふっ、おもろいなぁ二人とも", "クスクス笑う"),
        ("でもなぁ、現実的には……", "首を傾げる"),
        ("わたしはお茶でも淹れとくわ", "湯呑みを取り出す"),
        ("そういうとこ好きやで", "穏やかに微笑む"),
        ("ええもん見せてもろたわ", "目を細めて"),
        ("せやな、その通りやと思うで", "頷く"),
        ("ま、ぼちぼちいこか", "肩をすくめる"),
        ("う〜ん、それは賛成しかねるなぁ", "腕組み"),
        ("みんなで一緒やと安心するわ", "穏やかに笑う"),
    ],
    # フォールバック
    "_default": [
        ("そうだね", "頷く"),
        ("えっ、それは……", "首を傾げる"),
        ("いいんじゃない？", "微笑む"),
        ("うーん、どうしようかな", "腕組み"),
        ("そうかもね", "考え込む"),
        ("わかった、やってみる", "頷く"),
        ("ちょっと面白い", "目を細める"),
        ("……", "黙って聞く"),
    ],
}


def _detect_character_name(system_prompt: str) -> str:
    """system_prompt からキャラ名を抽出（プールを引くキー）.

    `あなたは「<NAME>」というキャラクターです` の <NAME> を最優先で取る。
    マッチしなければ単純な部分文字列検索でフォールバック。
    """
    # First-line speaker prompt pattern
    m = re.search(r"あなたは「?([぀-ヿ一-鿿ぁ-んァ-ン]{1,10})」?という", system_prompt)
    if m:
        cand = m.group(1)
        if cand in _RESPONSE_POOL:
            return cand

    for name in ("なでしこ", "千明", "あおい"):
        if name in system_prompt:
            return name

    return "_default"


def _detect_request_kind(system_prompt: str) -> str:
    """LLMRequest の種別を判定する.

    Priority:
        1. session_end : "記憶管理" / "episodic_memories" を含む
        2. emotion     : "感情分析" を含む（明示マーカー）
        3. summary     : "要約" / "summari" を含む
        4. chat        : それ以外（speech/thought/action を返す）

    Note: chat prompt は現在の PAD を提示するので "pleasure" / "arousal" の
    キーワードだけでは emotion と区別できない。emotion request は必ず
    "感情分析" / "記憶管理" 等の役割マーカーを system_prompt 先頭に持つ
    という前提に依存する。
    """
    if "記憶管理" in system_prompt or "episodic_memories" in system_prompt:
        return "session_end"
    if "感情分析" in system_prompt:
        return "emotion"
    if "要約" in system_prompt or "summari" in system_prompt.lower():
        return "summary"
    return "chat"


class MockLLMAdapter:
    """API キーなしで動かすための疑似 LLM."""

    def __init__(self, seed: int | None = None, latency_seconds: float = 0.0) -> None:
        self._rng = random.Random(seed)
        self._latency = latency_seconds
        self._call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self._call_count += 1
        if self._latency > 0:
            await asyncio.sleep(self._latency)

        kind = _detect_request_kind(request.system_prompt)

        if kind == "emotion":
            content = self._gen_emotion(request)
        elif kind == "session_end":
            content = self._gen_session_end()
        elif kind == "summary":
            content = "前回の会話の要約（mock）"
        else:
            content = self._gen_chat(request)

        # 適当な usage（実際の token 数とは異なる）
        usage = {
            "input_tokens": len(request.system_prompt) // 4
            + sum(len(m.get("content", "")) for m in request.messages) // 4,
            "output_tokens": len(content) // 4,
        }
        return LLMResponse(
            content=content,
            model="mock-llm",
            usage=usage,
        )

    def _gen_chat(self, request: LLMRequest) -> str:
        name = _detect_character_name(request.system_prompt)
        pool = _RESPONSE_POOL.get(name, _RESPONSE_POOL["_default"])
        speech, action = self._rng.choice(pool)

        # 「内心」っぽい thought を付ける
        thoughts = [
            "（みんな楽しそうだな）",
            "（次の話題どうしよう）",
            "（さっきの話、まだ気になる）",
            "（お腹減ったかも）",
            "（こういう時間、好きだな）",
            "",
        ]
        thought = self._rng.choice(thoughts)

        return json.dumps(
            {"speech": speech, "thought": thought, "action": action},
            ensure_ascii=False,
        )

    def _gen_emotion(self, request: LLMRequest) -> str:
        # 性格抽出は雑にやる: extraversion / neuroticism のキーワードから baseline 推定
        p = self._rng.uniform(-0.1, 0.6)
        a = self._rng.uniform(-0.2, 0.7)
        d = self._rng.uniform(-0.3, 0.5)

        # extraversion が prompt にあれば外向きに寄せる
        m = re.search(r"外向性[\s（(]*[Ee]xtraversion[）)]*[:：]\s*([0-9.]+)", request.system_prompt)
        if m:
            try:
                ext = float(m.group(1))
                p += (ext - 0.5) * 0.3
                a += (ext - 0.5) * 0.4
            except ValueError:
                pass

        p = max(-1.0, min(1.0, p))
        a = max(-1.0, min(1.0, a))
        d = max(-1.0, min(1.0, d))

        # emotion_label は雑に
        label = "happy" if p > 0.3 else ("sad_lite" if p < -0.1 else "neutral")
        situation = "雑談中"

        return json.dumps(
            {
                "pleasure": round(p, 2),
                "arousal": round(a, 2),
                "dominance": round(d, 2),
                "emotion_label": label,
                "situation": situation,
            },
            ensure_ascii=False,
        )

    def _gen_session_end(self) -> str:
        # 何もしないが空配列を返す（pipeline がエラーにならないように）
        return json.dumps(
            {
                "episodic_memories": [
                    {
                        "content": "今日は野クルの 3 人で楽しく雑談した",
                        "emotional_valence": 0.6,
                        "importance": 0.5,
                    },
                ],
                "semantic_updates": [],
                "user_context_updates": [],
                "relationship_changes": [],
            },
            ensure_ascii=False,
        )
