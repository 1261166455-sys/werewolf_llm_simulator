'''
LLM participation for the werewolf engine.

Everything that talks to a model lives in this file: the data structures handed
to a model, the description prompt, the provider adapters and the decoder that
turns a raw completion back into a move the engine will accept.  The game side
(ui.py) builds an Observation, calls LLMController.decide() and feeds the
resulting answer string back to engine.py - no part of engine.py knows about LLMs.

Data flow
---------
    GameState  --(ui.py curates, per viewer)-->  Observation
                                                     |
    engine prompt ("[Wolf] ... to kill")  -->  DecisionRequest
                                                     |
                                            build_messages() -> provider
                                                     |
                                   raw text -> decode() -> Decision
                                                     |
                                            Decision.as_answer() -> game_input

Only standard library HTTP is used, so no new dependency is required.
'''
from __future__ import annotations

import ast
import json
import os
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Protocol

import i18n

__all__ = [
    'DecisionKind', 'Observation', 'DecisionRequest', 'Decision',
    'classify', 'build_messages', 'decode', 'extract_json_object',
    'LLMError', 'DecodeError', 'ProviderConfig',
    'OpenAICompatProvider', 'AnthropicProvider', 'make_provider',
    'LLMController',
]


class LLMError(RuntimeError):
    '''The provider could not be reached or returned nothing usable.'''


class DecodeError(ValueError):
    '''A completion arrived but did not contain an acceptable decision.'''


class DecisionKind(Enum):
    '''What the engine is currently asking a player to do.'''
    Kill = 'kill'
    Save = 'save'
    Query = 'query'
    Vote = 'vote'
    Speech = 'speech'
    Generic = 'generic'


def classify(key: str, prompt: str) -> DecisionKind:
    '''
    Work out the kind of decision from what the engine exposes.

    engine.py only hands over a prompt string plus the `key` used by
    input_an_id(), so the mapping is by keyword.  If those prompts are ever
    reworded, this is the single place to update.
    '''
    if 'addresser' in prompt.lower():
        return DecisionKind.Speech
    if key == 'save':
        return DecisionKind.Save
    if key == 'kill':
        return DecisionKind.Kill
    text = prompt.lower()
    if 'to save' in text:
        return DecisionKind.Save
    if 'to kill' in text:
        return DecisionKind.Kill
    if 'query' in text:
        return DecisionKind.Query
    if 'vote' in text:
        return DecisionKind.Vote
    return DecisionKind.Generic


@dataclass(frozen=True)
class Observation:
    '''
    Everything one seat is allowed to know when it decides.

    This is the structure the prompt is rendered from, and it is deliberately
    written from a single player's point of view: the game side must fill it in
    per viewer, never hand over the raw GameState.
    '''
    viewer: int
    role: str
    stage: str
    cycle: int
    living: tuple[int, ...]
    out_of_play: tuple[int, ...]
    fallen_last_night: tuple[int, ...] = ()
    speeches: tuple[tuple[int, str], ...] = ()
    earlier_rounds: tuple[str, ...] = ()
    private_facts: tuple[str, ...] = ()
    vote_tally: tuple[tuple[int, int], ...] = ()

    def render(self, language: str | None = None) -> str:
        '''Human-readable public/private breakdown, used inside the user turn.'''
        lang = language or i18n.get_language()
        text = _labels(lang)
        stage = i18n.name(i18n.STAGES, self.stage, lang)
        lines = [
            text['public'],
            text['round'].format(cycle=self.cycle, stage=stage),
            text['alive'].format(ids=i18n.player_ids(self.living, lang)),
            text['out'].format(ids=i18n.player_ids(self.out_of_play, lang)),
            text['fallen'].format(ids=i18n.player_ids(self.fallen_last_night, lang)),
        ]
        if self.speeches:
            said = ' | '.join(text['said_item'].format(player=p, text=body)
                              for p, body in self.speeches)
            lines.append(text['said'].format(said=said))
        else:
            lines.append(text['said_none'])
        if self.vote_tally:
            separator = '，' if lang == 'zh' else ', '
            tally = separator.join(text['tally_item'].format(player=p, count=n)
                                   for p, n in self.vote_tally)
            lines.append(text['tally'].format(tally=tally))
        if self.earlier_rounds:
            lines.append(text['earlier'])
            lines.extend(f'    {r}' for r in self.earlier_rounds)
        lines.append(text['private'])
        if self.private_facts:
            lines.extend(f'- {fact}' for fact in self.private_facts)
        else:
            lines.append(text['private_none'])
        return '\n'.join(lines)


@dataclass(frozen=True)
class DecisionRequest:
    '''The engine's current question, in a form a model can be asked.'''
    kind: DecisionKind
    question: str
    allowed: tuple[int, ...] | None    # None -> free text (a public statement)
    required: bool                     # False -> passing is legal
    key: str = 'ID'


@dataclass
class Decision:
    '''A decoded, validated move.'''
    kind: DecisionKind
    choice: int | None = None
    speech: str = ''
    reasoning: str = ''
    raw: str = ''
    note: str = ''        # set when this is a fallback rather than a real reply

    def as_answer(self) -> str:
        '''The string engine.py expects from game_input()/input_an_id().'''
        if self.kind is DecisionKind.Speech:
            return self.speech
        return '' if self.choice is None else str(self.choice)


RULES: dict[str, str] = {'en': '''\
A six-player social-deduction game (simplified Werewolf). Seats are numbered 0-5.
Roles: 2 Wolves, 1 Prophet, 1 Witch, 2 Civilians, assigned at random.
The game runs up to 10 rounds; each round is one Night followed by one Day.

NIGHT, in this order:
1. Every living Wolf names one player to kill (a Wolf may pass). Wolves cannot
   name a Wolf. Every player named by any Wolf dies - if the Wolves disagree,
   more than one player dies.
2. The Prophet names one player and learns whether that player is a Wolf.
3. The Witch is shown the players killed by the Wolves. She may name one of
   those victims to kill, and may save one of those victims. Only that night's
   victims are legal targets for her.

DAY, in this order:
1. All deaths from the night are announced publicly.
2. Every living player makes one public statement, visible to everyone.
3. Every living player votes for a player, or gives up. The player with the most
   votes leaves the game; if the highest vote count is shared, nobody leaves.

WINNING:
- The civilians (Prophet, Witch and Civilians) win as soon as no Wolf is alive.
- The Wolves win as soon as no Prophet and no Witch are alive, or as soon as no
  Civilian is alive.
- If more than 10 rounds pass, the game is a draw.''',
    'zh': '''\
六人社交推理游戏（简化版狼人杀）。座位编号为 0-5。
身份：2 名狼人、1 名预言家、1 名女巫、2 名平民，随机分配。
最多进行 10 轮，每轮包含一个夜晚和一个白天。

夜晚按顺序进行：
1. 每位存活的狼人指定一名玩家击杀（可以跳过）。狼人不能选择狼人。
   被任意狼人指定的玩家都会死亡——如果狼人意见不一致，可能不止一人死亡。
2. 预言家指定一名玩家，得知其是否为狼人。
3. 女巫会看到被狼人击杀的玩家。她可以毒杀其中一名受害者，也可以救下其中一名。
   当晚她的合法目标只有这些受害者。

白天按顺序进行：
1. 公布夜晚的所有死亡。
2. 每位存活玩家发表一次公开陈述，所有人都能看到。
3. 每位存活玩家投票给一名玩家，或弃权。得票最多的玩家出局；
   如果最高票数并列，则无人出局。

胜负判定：
- 只要没有狼人存活，好人阵营（预言家、女巫、平民）立即获胜。
- 只要预言家和女巫都已不存活，或者所有平民都已不存活，狼人阵营立即获胜。
- 超过 10 轮仍未分出胜负则为平局。''',
}

ROLE_BRIEFING: dict[str, dict[str, str]] = {
    'en': {
    'Wolf': (
        'You are told which other player is a Wolf. Killing civilians and keeping '
        'suspicion off yourself is how you win. You may invent a role for yourself '
        'when you speak, but never contradict what your fellow Wolf says in public.'
    ),
    'Prophet': (
        'Each night you may name one player and learn whether that player is a '
        'Wolf; your results accumulate in your private notes. You win with the '
        'civilians, and revealing too much too early tends to get you killed at '
        'night - but staying silent lets the Wolves control the vote.'
    ),
    'Witch': (
        'Each night you are shown the players the Wolves killed; you may kill one '
        'of those victims and you may save one of them. You win with the '
        'civilians. Spending a save or a kill at the wrong moment can lose the '
        'game, so weigh whether the victim is likely to be an ally worth saving.'
    ),
    'Civilian': (
        'You have no night action and no secret knowledge. You win by reading the '
        'Wolves out of the information everyone shares: who died, who voted for '
        'whom, and who changes their story.'
    ),
    },
    'zh': {'Wolf': '你会得知另一名狼人是谁。击杀好人、并让自己不被怀疑，就是获胜的方式。发言时你可以伪装成其他身份，但不要和你的狼队友公开矛盾。', 'Prophet': '每晚你可以指定一名玩家，得知其是否为狼人；查验结果会累积在你的私人记录中。你与好人阵营一同获胜。过早暴露身份容易在夜里被杀，但一直沉默会让狼人掌控投票。', 'Witch': '每晚你会看到被狼人击杀的玩家；你可以毒杀其中一名受害者，也可以救下其中一名。你与好人阵营一同获胜。解药和毒药用错时机可能直接输掉游戏，所以要判断这名受害者是否值得救。', 'Civilian': '你没有夜间行动，也没有任何秘密信息。你只能依靠所有人都能看到的信息找出狼人：谁死了、谁投了谁、谁的发言前后矛盾。'},
}

CONTRACT: dict[str, str] = {'en': '''\
Reply with ONE JSON object and nothing else - no prose, no markdown fence:
{"choice": 3, "speech": "Player 3 has been quiet all day.", "reasoning": "..."}

"choice"     integer player id taken from the legal list, or null to pass when
             passing is allowed. Required for kill/save/query/vote decisions.
"speech"     the one public sentence to announce. Only used when you are asked
             for a statement; leave it as "" otherwise.
"reasoning"  one or two private sentences: why this move, who you suspect next.
             Never shown to the other players.
Answer with JSON only. If you cannot decide, still answer with JSON and choose
the pass option when it is legal.''',
    'zh': '''\
只回复一个 JSON 对象，不要输出任何其他内容，不要使用 markdown 代码块：
{"choice": 3, "speech": "玩家3今天一直不太说话。", "reasoning": "..."}

"choice"     从合法列表中选择的玩家编号；在允许放弃时也可以用 null 表示放弃。
             击杀/救人/查验/投票类问题必须给出编号。
"speech"     要公开发表的那一句话。只在需要发言时使用，其他情况留空字符串。
"reasoning"  一到两句私人推理：为什么这样选择、接下来怀疑谁。不会展示给其他玩家。
只输出 JSON。如果无法决定，也必须输出 JSON，并在允许时选择放弃。''',
}


LABELS: dict[str, dict[str, str]] = {
    'en': {
        'public': 'Public',
        'private': 'Private',
        'round': '- Round {cycle}, {stage}.',
        'alive': '- Alive: {ids}',
        'out': '- Out of play: {ids}',
        'fallen': '- Announced dead at the last dawn: {ids}',
        'said': '- Statements made today: {said}',
        'said_none': '- Statements made today: none yet',
        'said_item': 'P{player}: {text}',
        'tally': '- Votes cast so far today: {tally}',
        'tally_item': 'P{player}:{count}',
        'earlier': '- Earlier rounds:',
        'private_none': '- You hold no secrets beyond your role.',
        'seat_header': 'You are the decision maker for seat {viewer} in a game of Werewolf.',
        'section_seat': '# Your seat',
        'seat_line': 'You are Player {viewer}, the {role}.',
        'section_info': '# What you are told',
        'info_text': ('Public facts are known to everyone. Private facts are known only to '
                      'you: never state them verbatim in a public statement, though you may '
                      'hint at them when it helps you win.'),
        'section_answer': '# Your answer',
        'question_header': 'Question',
        'question_kind': '- kind: {kind}',
        'question_text': '- {question}',
        'legal': '- legal choices: {legal}',
        'legal_free': 'free text - you may say anything (one short sentence)',
        'legal_none': 'no legal choice exists',
        'legal_pass': '  (or null to pass)',
        'corrective': ('Your previous reply was rejected: {error}\n'
                       'Previous reply was: {previous}\n'
                       'Answer again with a single valid JSON object.'),
        'kind_kill': 'kill', 'kind_save': 'save', 'kind_query': 'query',
        'kind_vote': 'vote', 'kind_speech': 'speech', 'kind_generic': 'generic',
    },
    'zh': {
        'public': '公开信息',
        'private': '私人信息',
        'round': '- 第 {cycle} 轮，{stage}。',
        'alive': '- 存活：{ids}',
        'out': '- 已出局：{ids}',
        'fallen': '- 昨夜公布的死亡：{ids}',
        'said': '- 今日发言：{said}',
        'said_none': '- 今日发言：暂无',
        'said_item': '玩家{player}：{text}',
        'tally': '- 当前票数：{tally}',
        'tally_item': '玩家{player} {count} 票',
        'earlier': '- 往轮情况：',
        'private_none': '- 除你的身份外，你没有其他秘密信息。',
        'seat_header': '你要为狼人杀游戏中的 {viewer} 号座位做决策。',
        'section_seat': '# 你的身份',
        'seat_line': '你是 {viewer} 号玩家，身份是{role}。',
        'section_info': '# 你会获得的信息',
        'info_text': ('公开信息所有人都知道。私人信息只有你知道：不要在任何公开陈述中原文照搬，'
                      '但在有助于你获胜时可以暗示。'),
        'section_answer': '# 回答格式',
        'question_header': '问题',
        'question_kind': '- 类型：{kind}',
        'question_text': '- {question}',
        'legal': '- 合法选择：{legal}',
        'legal_free': '自由文本——你可以说任意一句话（一句话即可）',
        'legal_none': '没有任何合法选择',
        'legal_pass': '（也可以用 null 表示放弃）',
        'corrective': ('你上一次的回复被拒绝：{error}\n'
                       '上一次的回复是：{previous}\n'
                       '请重新只输出一个合法的 JSON 对象。'),
        'kind_kill': '击杀', 'kind_save': '救人', 'kind_query': '查验',
        'kind_vote': '投票', 'kind_speech': '发言', 'kind_generic': '行动',
        'q_kill': '你今晚要击杀哪名玩家？',
        'q_save': '你要救下哪名玩家？',
        'q_query': '你要查验哪名玩家？',
        'q_vote': '你要投票给谁？',
        'q_speech': '请发表你的公开陈述。',
    },
}


def _labels(language: str | None = None) -> dict[str, str]:
    return LABELS.get(language or i18n.get_language()) or LABELS['en']


def build_messages(
    observation: Observation,
    request: DecisionRequest,
    corrective: str = '',
    previous: str = '',
    language: str | None = None,
) -> tuple[str, str]:
    '''
    Render the description prompt.

    Returns (system, user). Kept as a pure function so the exact wording can be
    inspected, unit tested and tuned without touching any transport code. The
    English output is deliberately byte-for-byte what earlier verification ran
    against, so only the Chinese path is new.
    '''
    lang = language or i18n.get_language()
    text = _labels(lang)
    role = i18n.name(i18n.JOB_NAMES, observation.role, lang)
    briefing = (ROLE_BRIEFING.get(lang) or ROLE_BRIEFING['en']).get(observation.role, '')

    system = '\n\n'.join([
        text['seat_header'].format(viewer=observation.viewer),
        RULES.get(lang) or RULES['en'],
        '\n'.join([
            text['section_seat'],
            text['seat_line'].format(viewer=observation.viewer, role=role),
            briefing,
        ]),
        '\n'.join([text['section_info'], text['info_text']]),
        '\n'.join([text['section_answer'], CONTRACT.get(lang) or CONTRACT['en']]),
    ])

    if request.allowed is None:
        legal = text['legal_free']
    else:
        if lang == 'zh':
            legal = '、'.join(str(i) for i in request.allowed) or text['legal_none']
        else:
            legal = ', '.join(str(i) for i in request.allowed) or text['legal_none']
        if not request.required:
            legal += text['legal_pass']

    # engine.py's own prompt is English, so the Chinese prompt states the question
    # from the classified kind instead of echoing it.
    if lang == 'zh':
        question = text.get(f'q_{request.kind.value}') or request.question
    else:
        question = request.question

    user = '\n'.join([
        observation.render(lang),
        text['question_header'],
        text['question_kind'].format(kind=text.get(f'kind_{request.kind.value}', request.kind.value)),
        text['question_text'].format(question=question.strip()),
        text['legal'].format(legal=legal),
    ])

    if corrective:
        user += '\n\n' + text['corrective'].format(error=corrective, previous=previous[:400])
    return system, user


# -- decoding --------------------------------------------------------------
_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.DOTALL)
_PLAYER_ID = re.compile(r'^(?:player|p|seat)?\s*#?\s*(\d+)$', re.IGNORECASE)
_PASS_WORDS = {'', 'null', 'none', 'skip', 'pass', 'give up', 'giveup', 'no one', 'nobody', '-',
               # Chinese equivalents, since a Chinese prompt invites Chinese answers
               '放弃', '跳过', '弃权', '不投票', '不选', '无', '没有', '无人', '空'}


_CONTROL_ESCAPE = {'\n': '\\n', '\r': '\\r', '\t': '\\t'}


def _repair_json(text: str) -> str:
    '''
    Fix the two mistakes models make most often, both of which make json.loads
    throw away an otherwise usable reply: raw control characters inside a string
    (a literal newline in a multi-line "reasoning") and trailing commas.
    '''
    chars: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            elif char in _CONTROL_ESCAPE:
                chars.append(_CONTROL_ESCAPE[char])
                continue
            chars.append(char)
            continue
        if char == '"':
            in_string = True
        chars.append(char)
    return re.sub(r',(\s*[}\]])', r'\1', ''.join(chars))


def extract_json_object(text: str) -> dict[str, Any]:
    '''
    Pull the first JSON object out of a completion.

    Models wrap JSON in fences, prepend apologies or append commentary, so try
    the fenced block first, then brace matching over the raw text, then the same
    candidates after repair. A Python-style dict is accepted as a last resort
    because some models emit single quotes despite being asked for JSON.
    '''
    if not text or not text.strip():
        raise DecodeError('empty completion')

    candidates: list[str] = []
    for match in _FENCE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())
    braced = _first_balanced_object(text)
    if braced:
        candidates.append(braced)

    for candidate in candidates:
        for form in (candidate, _repair_json(candidate)):
            try:
                parsed = json.loads(form)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed[0]

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(_repair_json(candidate))
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed

    raise DecodeError(f'no JSON object found in completion: {text[:200]!r}')


def _first_balanced_object(text: str) -> str | None:
    '''Scan for a {...} span, ignoring braces inside strings.'''
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def coerce_choice(value: Any) -> int | None:
    '''
    Turn whatever the model wrote into a player id or None.

    Accepts 3, "3", "Player 3", "P3", "seat #3", null, "pass", "skip" ...
    '''
    if value is None:
        return None
    if isinstance(value, bool):
        raise DecodeError(f'cannot use a boolean as a player id: {value!r}')
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise DecodeError(f'cannot use {value!r} as a player id')
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _PASS_WORDS:
            return None
        match = _PLAYER_ID.match(text)
        if match:
            return int(match.group(1))
    raise DecodeError(f'cannot read a player id from {value!r}')


def decode(raw: str, request: DecisionRequest) -> Decision:
    '''
    Validate a completion against the question that was asked.

    Raises DecodeError for anything the engine would reject, so the controller
    can re-ask instead of letting engine.py loop on a bad answer.
    '''
    payload = extract_json_object(raw)

    speech = payload.get('speech') or ''
    if not isinstance(speech, str):
        speech = str(speech)
    reasoning = payload.get('reasoning') or ''
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    if request.kind is DecisionKind.Speech:
        if not speech.strip():
            raise DecodeError('the statement ("speech") was empty')
        return Decision(DecisionKind.Speech, None, speech.strip(), reasoning.strip(), raw)

    if 'choice' not in payload:
        raise DecodeError('the JSON object has no "choice" field')

    choice = coerce_choice(payload['choice'])

    if choice is None:
        if request.required:
            raise DecodeError('null was returned but this question requires a target')
        return Decision(request.kind, None, speech.strip(), reasoning.strip(), raw)

    if request.allowed is not None and choice not in request.allowed:
        raise DecodeError(f'{choice} is not a legal target; legal choices are {list(request.allowed)}')

    return Decision(request.kind, choice, speech.strip(), reasoning.strip(), raw)


# -- providers -------------------------------------------------------------
DEFAULT_BASE_URLS = {
    'deepseek': 'https://api.deepseek.com/v1',
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com',
    'compatible': '',        # any other OpenAI-shaped gateway: supply base_url
}
DEFAULT_MODELS = {
    'deepseek': 'deepseek-chat',
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-5-haiku-latest',
    'compatible': '',
}
KEY_ENV = {
    'deepseek': 'DEEPSEEK_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'compatible': 'LLM_API_KEY',
}


@dataclass
class ProviderConfig:
    '''Connection settings. The API key is never included in public_state().'''
    kind: str = 'deepseek'
    base_url: str = ''
    model: str = ''
    api_key: str = ''
    temperature: float = 0.8
    max_tokens: int = 400
    timeout: float = 60.0
    json_mode: bool = False       # ask for response_format=json_object when supported

    def __post_init__(self) -> None:
        self.kind = (self.kind or 'deepseek').strip().lower()
        if self.kind not in DEFAULT_BASE_URLS:
            raise LLMError(
                f'unknown provider kind {self.kind!r}; expected one of '
                + ', '.join(sorted(DEFAULT_BASE_URLS))
            )
        self.base_url = (self.base_url or DEFAULT_BASE_URLS[self.kind]).rstrip('/')
        if not self.base_url:
            raise LLMError(f'kind {self.kind!r} needs an explicit base_url')
        self.model = self.model or DEFAULT_MODELS[self.kind]
        if not self.model:
            raise LLMError(f'kind {self.kind!r} needs an explicit model')

    @classmethod
    def from_env(cls) -> 'ProviderConfig':
        kind = os.getenv('LLM_KIND', 'deepseek').strip().lower()
        key = os.getenv('LLM_API_KEY') or os.getenv(KEY_ENV.get(kind) or 'LLM_API_KEY', '')
        return cls(
            kind=kind,
            base_url=os.getenv('LLM_BASE_URL', ''),
            model=os.getenv('LLM_MODEL', ''),
            api_key=key or '',
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.8')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '400')),
            json_mode=os.getenv('LLM_JSON_MODE', '').lower() in {'1', 'true', 'yes'},
        )

    def describe(self) -> dict[str, Any]:
        '''Browser-safe view: says whether a key exists, never what it is.'''
        return {
            'kind': self.kind,
            'base_url': self.base_url,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'json_mode': self.json_mode,
            'has_key': bool(self.api_key),
        }


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str:
        '''Return the raw assistant text for one turn.'''


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', **headers},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', 'replace')[:400]
        raise LLMError(f'HTTP {error.code} from {url}: {detail}') from error
    except urllib.error.URLError as error:
        raise LLMError(f'cannot reach {url}: {error.reason}') from error
    except socket.timeout as error:
        raise LLMError(f'timed out after {timeout}s waiting for {url}') from error
    except json.JSONDecodeError as error:
        raise LLMError(f'{url} returned a body that is not JSON') from error


class OpenAICompatProvider:
    '''
    POST /chat/completions. This is the default transport: DeepSeek and OpenAI
    both speak it, as do Qwen, GLM, Moonshot, Ollama, vLLM, LM Studio and any
    other gateway selected with the 'compatible' kind.
    '''
    name = 'openai-compatible'

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise LLMError(
                'no API key configured - set '
                f'{KEY_ENV.get(config.kind, "LLM_API_KEY")} (or LLM_API_KEY), '
                'or paste one into the console before starting'
            )
        self.config = config

    def complete(self, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            'model': self.config.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
        }
        if self.config.json_mode:
            payload['response_format'] = {'type': 'json_object'}

        data = _post_json(
            f'{self.config.base_url}/chat/completions',
            payload,
            {'Authorization': f'Bearer {self.config.api_key}'},
            self.config.timeout,
        )
        try:
            return data['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f'unexpected completion shape: {json.dumps(data)[:300]}') from error


class AnthropicProvider:
    name = 'anthropic'

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise LLMError(
                'no API key configured - set ANTHROPIC_API_KEY (or LLM_API_KEY)'
            )
        self.config = config

    def complete(self, system: str, user: str) -> str:
        payload = {
            'model': self.config.model,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'system': system,
            'messages': [{'role': 'user', 'content': user}],
        }
        data = _post_json(
            f'{self.config.base_url}/v1/messages',
            payload,
            {'x-api-key': self.config.api_key, 'anthropic-version': '2023-06-01'},
            self.config.timeout,
        )
        try:
            blocks = [b['text'] for b in data['content'] if b.get('type') == 'text']
        except (KeyError, TypeError) as error:
            raise LLMError(f'unexpected completion shape: {json.dumps(data)[:300]}') from error
        return ''.join(blocks)


def make_provider(config: ProviderConfig) -> Provider:
    return AnthropicProvider(config) if config.kind == 'anthropic' else OpenAICompatProvider(config)


# -- controller ------------------------------------------------------------
class LLMController:
    '''
    Turns (Observation, DecisionRequest) into a Decision the engine will accept.

    Legality is enforced here rather than trusted from the model: a bad target
    would make engine.py re-prompt forever, so an unusable reply is retried once
    and then replaced by a legal fallback.
    '''

    def __init__(self, provider: Provider, attempts: int = 2, on_event=None,
                 language: str | None = None) -> None:
        self.provider = provider
        self.attempts = max(1, attempts)
        self.on_event = on_event
        self.language = language

    def _emit(self, **event: Any) -> None:
        if self.on_event is not None:
            self.on_event(event)

    def decide(self, observation: Observation, request: DecisionRequest) -> Decision:
        failure = ''
        previous = ''
        for attempt in range(1, self.attempts + 1):
            system, user = build_messages(
                observation, request,
                corrective=failure if attempt > 1 else '',
                previous=previous,
                language=self.language,
            )
            self._emit(kind='llm-request', viewer=observation.viewer, attempt=attempt,
                           system=system, user=user)
            try:
                raw = self.provider.complete(system, user)
            except LLMError as error:
                failure = str(error)
                self._emit(kind='llm-error', viewer=observation.viewer, attempt=attempt, text=failure)
                continue

            self._emit(kind='llm-response', viewer=observation.viewer, attempt=attempt, text=raw)
            try:
                decision = decode(raw, request)
            except DecodeError as error:
                failure = str(error)
                previous = raw
                self._emit(kind='llm-error', viewer=observation.viewer, attempt=attempt, text=failure)
                continue
            return decision

        return self.fallback(request, failure, self.language)

    @staticmethod
    def fallback(request: DecisionRequest, failure: str,
                 language: str | None = None) -> Decision:
        '''Never hand the engine something it will reject.'''
        lang = language or i18n.get_language()
        note = i18n.t('fallback_note', lang, error=failure)
        if request.kind is DecisionKind.Speech:
            return Decision(request.kind, None, i18n.t('fallback_speech', lang), note=note)
        pick = request.allowed[0] if request.required and request.allowed else None
        return Decision(request.kind, pick, note=note)
