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


# キャラ名 → 思惑（surface_goal, hidden_goal, intensity）プール (Issue #20).
# intent-emergent-not-scripted を mock でも担保するため複数案からランダムに選ぶ。
_INTENT_POOL: dict[str, list[tuple[str, str, float]]] = {
    "なでしこ": [
        ("みんなで美味しいご飯を食べる流れにしたい", "褒められたくてうずうずしてる", 0.7),
        ("次のキャンプの話で盛り上げたい", "本当はただ構ってほしいだけ", 0.6),
        ("自分の好きなものをみんなに気に入ってほしい", "", 0.65),
    ],
    "千明": [
        ("自分の企画をこの場で通したい", "本当はなでしこに負けたくないだけ", 0.85),
        ("会話の主導権を握ってリードしたい", "リーダーとして認められたい", 0.8),
        ("みんなを巻き込んで無茶な計画を実現したい", "", 0.75),
    ],
    "あおい": [
        ("対立しそうな空気を穏やかに収めたい", "本当は自分が一番気疲れしたくない", 0.6),
        ("みんなが仲良くいられる落としどころを探したい", "", 0.55),
        ("二人を見守りつつ場の安全を保ちたい", "千明の暴走をやんわり止めたい", 0.5),
    ],
    "_default": [
        ("この会話で自分の考えを伝えたい", "", 0.5),
        ("場の空気を心地よく保ちたい", "", 0.45),
    ],
}


def _default_payload_for_schema(schema: dict) -> dict:
    """Build a minimal dict satisfying the JSON Schema's required fields (Issue #12)."""
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    out: dict = {}
    for field_name in required:
        spec = props.get(field_name, {}) or {}
        ftype = spec.get("type")
        if ftype == "number" or ftype == "integer":
            out[field_name] = 0.0 if ftype == "number" else 0
        elif ftype == "string":
            out[field_name] = ""
        elif ftype == "array":
            out[field_name] = []
        elif ftype == "object":
            out[field_name] = {}
        elif ftype == "boolean":
            out[field_name] = False
        else:
            out[field_name] = None
    return out


def _detect_character_name(system_prompt: str) -> str:
    """system_prompt からキャラ名を抽出（プールを引くキー）.

    優先順位:
        1. `あなたは「<NAME>」というキャラクターです`（発話プロンプト）の <NAME>。
        2. `名前: <NAME>`（思惑生成プロンプト, Issue #20）の <NAME>。
        3. 単純な部分文字列検索（profile が他キャラ名を含むと誤検出しうるので最後）。

    部分文字列フォールバックを最後に置く理由: あおいの profile は「千明の暴走を
    いなす」のように他キャラ名を含むため、構造化されたマーカー行を優先しないと
    別キャラのプールを引いてしまう。
    """
    # 1. 発話プロンプト「あなたは「<NAME>」という」
    m = re.search(r"あなたは「?([぀-ヿ一-鿿ぁ-んァ-ン]{1,10})」?という", system_prompt)
    if m:
        cand = m.group(1)
        if cand in _RESPONSE_POOL:
            return cand

    # 2. 思惑生成プロンプト「名前: <NAME>」
    m = re.search(r"名前[:：]\s*([぀-ヿ一-鿿ぁ-んァ-ン]{1,10})", system_prompt)
    if m:
        cand = m.group(1)
        if cand in _RESPONSE_POOL:
            return cand

    # 3. 部分文字列フォールバック
    for name in ("なでしこ", "千明", "あおい"):
        if name in system_prompt:
            return name

    return "_default"


# Session が speech プロンプトに思惑を注入するときのマーカー (Issue #20).
# mock がこの行を拾って thought に本音を反映する。
_SURFACE_MARKER = "表のゴール:"
_HIDDEN_MARKER = "裏の本音:"


def _extract_injected_marker(system_prompt: str, marker: str) -> str:
    """system_prompt 中の `<marker> <値>` 行から値を取り出す（無ければ ""）."""
    idx = system_prompt.find(marker)
    if idx == -1:
        return ""
    rest = system_prompt[idx + len(marker):]
    line = rest.split("\n", 1)[0].strip()
    if line in ("（なし）", "(none)", "なし"):
        return ""
    return line


def _extract_injected_surface_goal(system_prompt: str) -> str:
    return _extract_injected_marker(system_prompt, _SURFACE_MARKER)


def _extract_injected_hidden_goal(system_prompt: str) -> str:
    return _extract_injected_marker(system_prompt, _HIDDEN_MARKER)


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

    # Issue #12: structured_completion path is supported by mock, so emotion_engine
    # and session_end pipelines exercise the same code branch as real Claude.
    supports_structured_completion: bool = True

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

    async def structured_completion(
        self,
        *,
        system_prompt: str,
        messages: list[dict],
        tool_name: str,
        tool_description: str,
        schema: dict,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> dict:
        """Tool-Use 模倣: schema に対応した dict を返す (Issue #12).

        tool_name が `report_emotion` なら emotion 用 dict、
        `record_session_memories` なら session_end 用 dict、
        それ以外は schema の required を埋めた最小の dict を返す.

        invariant: structured-output-via-tool-use
        """
        self._call_count += 1
        if self._latency > 0:
            await asyncio.sleep(self._latency)

        if tool_name == "report_emotion":
            request = LLMRequest(system_prompt=system_prompt, messages=messages)
            return json.loads(self._gen_emotion(request))
        if tool_name == "record_session_memories":
            return json.loads(self._gen_session_end())
        if tool_name == "report_intent":
            return self._gen_intent(system_prompt)

        # Generic fallback: 各 required を最小値で埋める
        return _default_payload_for_schema(schema)

    def _gen_chat(self, request: LLMRequest) -> str:
        name = _detect_character_name(request.system_prompt)
        pool = _RESPONSE_POOL.get(name, _RESPONSE_POOL["_default"])
        speech, action = self._rng.choice(pool)

        # speech-thought-separation (Issue #20):
        # system_prompt に思惑（hidden_goal）が注入されていれば、thought は
        # その本音を反映する（表の speech とギャップが出る）。
        thought = self._gen_thought(request.system_prompt)

        return json.dumps(
            {"speech": speech, "thought": thought, "action": action},
            ensure_ascii=False,
        )

    def _gen_thought(self, system_prompt: str) -> str:
        """思惑（特に hidden_goal）を反映した内心を生成する (Issue #20)."""
        hidden = _extract_injected_hidden_goal(system_prompt)
        if hidden:
            return f"（{hidden}……それは言えないけど）"
        surface = _extract_injected_surface_goal(system_prompt)
        if surface:
            return f"（{surface}……そこに持っていきたいな）"
        thoughts = [
            "（みんな楽しそうだな）",
            "（次の話題どうしよう）",
            "（さっきの話、まだ気になる）",
            "（お腹減ったかも）",
            "（こういう時間、好きだな）",
            "",
        ]
        return self._rng.choice(thoughts)

    def _gen_intent(self, system_prompt: str) -> dict:
        """report_intent tool 用の構造化思惑を返す (Issue #20).

        invariant: intent-emergent-not-scripted — 同じキャラでも複数案から
        ランダム選択して、思惑の展開が決定論的にならないようにする。
        """
        name = _detect_character_name(system_prompt)
        pool = _INTENT_POOL.get(name, _INTENT_POOL["_default"])
        surface, hidden, intensity = self._rng.choice(pool)
        return {
            "surface_goal": surface,
            "hidden_goal": hidden,
            "intensity": intensity,
        }

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
