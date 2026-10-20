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
from pathlib import Path
from enum import Enum
from typing import Any, Iterable, Protocol

import i18n

__all__ = [
    'DecisionKind', 'Observation', 'DecisionRequest', 'Decision',
    'classify', 'build_messages', 'decode', 'extract_json_object',
    'LLMError', 'BudgetExhausted', 'DecodeError', 'ProviderConfig',
    'OpenAICompatProvider', 'AnthropicProvider', 'make_provider',
    'LLMController', 'ModelSpec', 'ModelRegistry', 'parse_model_config',
    'load_model_config', 'MODEL_FIELDS',
]


class LLMError(RuntimeError):
    '''The provider could not be reached or returned nothing usable.'''


class BudgetExhausted(LLMError):
    '''
    The reply was cut off before any content arrived.

    A thinking model spends the budget on its chain of thought first, so a small
    `max_tokens` yields finish_reason="length" with an empty message. That is not
    the model's fault and retrying unchanged cannot help: the caller widens the
    budget and asks again.
    '''


class DecodeError(ValueError):
    '''A completion arrived but did not contain an acceptable decision.'''


class DecisionKind(Enum):
    '''What the engine is currently asking a player to do.'''
    Kill = 'kill'                 # 女巫 poison
    PackKill = 'pack_kill'        # 狼队 collective proposal
    Save = 'save'                 # 女巫 antidote
    Query = 'query'               # 预言家
    Guard = 'guard'               # 守卫
    Vote = 'vote'                 # 放逐投票
    Speech = 'speech'             # 发言 / 竞选发言
    Shot = 'shot'                 # 猎人 / 狼王 开枪
    Badge = 'badge'               # 是否移交警徽 (yes/no)
    BadgeTarget = 'badge_target'  # 警徽交给谁 (an id)
    Police = 'police'             # 是否上警
    Expose = 'expose'             # 狼人自爆
    SpeechOrder = 'speech_order'  # 警长决定发言顺序
    Generic = 'generic'


# Questions answerable with yes/no. The engine reads a string, and a blank
# answer means "no" everywhere, so these decode to 'y' / 'n'.
YES_NO_KINDS = frozenset({DecisionKind.Police, DecisionKind.Expose,
                          DecisionKind.Badge, DecisionKind.SpeechOrder})


def classify(key: str, prompt: str) -> DecisionKind:
    '''
    Work out the kind of decision from what the engine exposes.

    engine.py only hands over a prompt string plus the `key` used by
    input_an_id(), so the mapping is by keyword, with `key` checked first
    because it is the stabler of the two. If those prompts are ever reworded,
    this is the single place to update - and tests/test_bridge.py asserts that
    every prompt a real game produces classifies to something specific.
    '''
    text = prompt.lower()
    if 'addresser' in text or 'speecher' in text:
        return DecisionKind.Speech
    # The pack's proposal is not the witch's poison: keeping them apart stops a
    # Chinese wolf from being asked which player it wants to poison.
    if 'propose a player to kill' in text:
        return DecisionKind.PackKill
    if key in ('target', 'guard'):
        return DecisionKind.Guard
    if key == 'save':
        return DecisionKind.Save
    if key == 'kill':
        return DecisionKind.Kill
    if 'expose your identification' in text:
        return DecisionKind.Expose
    if 'choose to be the police' in text:
        return DecisionKind.Police
    if 'comment order' in text:
        return DecisionKind.SpeechOrder
    # Two different questions mention the badge: "do you want to hand it on?"
    # is yes/no, "which player?" takes an id. Treating them as one kind made the
    # console offer 是/否 buttons for an id question, and the engine then looped
    # on "invalid format" because 'y' is not a seat number.
    if 'transmit your badge' in text:
        return DecisionKind.Badge
    if 'send your badge' in text or 'send a badge' in text:
        return DecisionKind.BadgeTarget
    if 'badge' in text:
        return DecisionKind.Badge
    if 'shoot' in text:
        return DecisionKind.Shot
    if 'to save' in text:
        return DecisionKind.Save
    if 'to kill' in text:
        return DecisionKind.Kill
    if 'to guard' in text:
        return DecisionKind.Guard
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
    # public political state: the sheriff and the election
    sheriff: int | None = None
    sheriff_badge_broken: bool = False
    sheriff_candidates: tuple[int, ...] = ()
    candidate_speeches: tuple[tuple[int, str], ...] = ()
    exploded_wolves: int = 0

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
        if self.sheriff is not None:
            lines.append(text['sheriff'].format(sheriff=self.sheriff))
        else:
            lines.append(text['sheriff_none'])
        if self.sheriff_badge_broken:
            lines.append(text['sheriff_broken'])
        if self.sheriff_candidates:
            lines.append(text['candidates'].format(
                ids=i18n.player_ids(self.sheriff_candidates, lang)))
        if self.candidate_speeches:
            said = ' | '.join(text['said_item'].format(player=p, text=body)
                              for p, body in self.candidate_speeches)
            lines.append(text['candidate_said'].format(said=said))
        if self.exploded_wolves:
            lines.append(text['exploded'].format(count=self.exploded_wolves))
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
    yes: bool | None = None   # only for a yes/no question

    def as_answer(self) -> str:
        '''
        The string engine.py expects.

        game_input()/input_an_id() read plain strings: an id, a statement, or
        'y'/'n'. A blank string means "skip", which for every yes/no question
        the engine asks is the safe "no" (it tears the badge, does not explode).
        '''
        if self.kind is DecisionKind.Speech:
            return self.speech
        if self.kind in YES_NO_KINDS:
            return 'y' if self.yes else 'n'
        return '' if self.choice is None else str(self.choice)


RULES: dict[str, str] = {'en': """\
A twelve-player social-deduction game (Werewolf). Seats are numbered 0-11.
Roles: 3 Wolves, 1 Wolf King, 1 Prophet, 1 Witch, 1 Guard, 1 Hunter, 4 Civilians.
The game runs up to 10 rounds; each round is one Night followed by one Day.

NIGHT, in this order:
1. Guard: names one player to protect, or passes. He may protect himself. The
   same player may not be protected two nights in a row. A protected player
   survives the Wolves' kill, but NOT the Witch's poison.
2. Wolves (including the Wolf King): the pack discusses in seat order and agrees
   on at most one victim. Each living Wolf proposes one living player, or passes,
   and sees the proposals already made tonight. The most-proposed player dies; a
   tie, or every Wolf passing, means nobody dies. A Wolf may propose a packmate
   or himself.
3. Prophet: names one living player other than herself and learns that player's
   camp (good or wolf). The Wolf King reads as a wolf.
4. Witch: she is shown tonight's victims. She may use her poison once per game
   on any living player except herself, and her antidote once per game on one of
   tonight's victims (she may save herself from the second night on). A player
   killed by poison cannot be saved by anything.
   If the Witch saves a player who was also protected by the Guard, that player
   dies anyway (the potion and the protection cancel out).

DAY, in this order:
1. The night's deaths are announced; those players leave the game.
2. The game may end here.
3. On the first day only: the players elect a sheriff (see below), and the
   players who died on the first night give a last statement before leaving.
4. Every living player makes one public statement, in an order the sheriff
   chooses (from himself, ascending or descending seat order).
   A Wolf may, when it is his turn, declare himself a Wolf and leave the game at
   once (self-explosion). His statement stands; every later statement that day
   and the whole vote are skipped, and the game goes straight to the next night.
5. Every living player votes for a player, or passes. The player with the most
   votes leaves; a tie means nobody leaves.
6. The game is judged again, then the next night begins.

SHERIFF: elected on the first day. Candidates speak, everyone else votes; the
candidate with the most votes becomes sheriff, and a tie or no candidate means
no sheriff this game. The sheriff's vote counts 1.5. When the sheriff leaves the
game he hands the badge to a living player of his choice, or tears it. If two
Wolves have exploded, the badge is discarded.

SHOTS: the Hunter may shoot one living player when he is voted out or killed by
the Wolves (not when poisoned). The Wolf King may shoot one living player when he
is voted out, and never when he is killed at night.

WINNING:
- The good side (Prophet, Witch, Guard, Hunter, Civilians) wins when no Wolf is alive.
- The Wolves win when every god (Prophet, Witch, Guard, Hunter) is out, or every
  Civilian is out.
- If more than 10 rounds pass, the game is a draw.""",
    'zh': """\
十二人社交推理游戏（狼人杀）。座位编号为 0-11。
身份：3 名狼人、1 名狼王、1 名预言家、1 名女巫、1 名守卫、1 名猎人、4 名平民，随机分配。
最多进行 10 轮，每轮包含一个夜晚和一个白天。

夜晚按顺序进行：
1. 守卫：守护一名玩家或放弃，可以守护自己；同一名玩家不能连续两晚被守护。
   被守护者可以免疫狼刀，但不能免疫女巫的毒药。
2. 狼人（包括狼王）：狼队按座位顺序商议，最多击杀一名玩家。每位存活狼人提议一名
   存活玩家或放弃，并能看到今晚此前队友的提议。被提议最多的玩家死亡；票数并列或
   全部放弃则无人死亡。狼人可以提议同阵营的狼人或自己。
3. 预言家：指定一名自己之外的存活玩家，得知其阵营（好人/狼人）。狼王属于狼人阵营。
4. 女巫：会看到今晚被狼人击杀的玩家。全局只有一瓶毒药和一瓶解药：
   毒药可以对任意非己存活玩家使用；解药只能救今晚被狼杀的玩家，从第二晚起可以自救。
   被毒死的玩家无法被任何方式救回。
   若女巫救下的玩家同时被守卫守护，该玩家仍然死亡（同守同救）。

白天按顺序进行：
1. 公布昨夜死亡名单，这些玩家出局。
2. 此时可能直接判定胜负。
3. 仅第一天：进行警长竞选；第一天夜晚死亡的玩家发表遗言。
4. 每位存活玩家按顺序发表一次公开陈述，顺序由警长决定（从警长开始，按座位正序或逆序）。
   狼人可以在自己发言时自爆身份并立即出局：他的发言保留，当天之后的所有发言与投票都被跳过，
   游戏直接进入下一夜晚。
5. 每位存活玩家投票给一名玩家或弃权。得票最多者出局；最高票并列则无人出局。
6. 再次判定胜负，然后进入下一天夜晚。

警长：第一天竞选产生。竞选者发言，其余玩家投票；得票最多的竞选者成为警长，
平票或无人竞选则本局无警长。警长的票计 1.5 票。警长出局时可以把警徽移交给一名存活玩家，
或者撕毁；若已有两只狼自爆，警徽会被强制废弃。

开枪：猎人被投票放逐或被狼刀死时可以开枪带走一名存活玩家（被毒死不能开枪）。
狼王仅在被投票放逐时可以开枪，夜里被刀死不能开枪。

胜负判定：
- 没有狼人存活时，好人阵营（预言家、女巫、守卫、猎人、平民）获胜。
- 所有神职（预言家、女巫、守卫、猎人）出局，或所有平民出局时，狼人阵营获胜。
- 超过 10 轮仍未分出胜负则为平局。""",
}

ROLE_BRIEFING: dict[str, dict[str, str]] = {
    'en': {
        'Wolf': (
            "You are told who the rest of the pack is, including the Wolf King. "
            "Every night the pack discusses in seat order: each of you proposes "
            "one living player, and the most-proposed player dies - an even split "
            "kills nobody, so a scattered pack wastes the night. If you speak "
            "later you see what your packmates proposed. Killing the good side and "
            "keeping suspicion off yourself is how you win. You may claim any role "
            "when you speak, but never contradict a packmate in public. During "
            "your own statement you may also declare yourself a Wolf: that ends "
            "the day at once and costs you your life, so only do it to stop a vote "
            "that would kill a packmate."
        ),
        'Wolf King': (
            "You hunt with the pack and know your packmates. If you are voted out "
            "you may shoot one living player as you die - use it to take a god "
            "with you. You cannot shoot when you are killed at night."
        ),
        'Prophet': (
            "Each night you name one living player other than yourself and learn "
            "that player's camp; results accumulate in your private notes. The "
            "Wolf King reads as a wolf. You win with the good side. Claiming too "
            "early gets you killed at night; staying silent lets the Wolves steer "
            "the vote."
        ),
        'Witch': (
            "You hold one poison and one antidote for the whole game. You are shown "
            "tonight's victims: you may poison any living player except yourself, "
            "and you may save one of tonight's victims (from the second night you "
            "may save yourself). Poison always wins: a poisoned player cannot be "
            "saved, and a player you save who was also guarded dies anyway."
        ),
        'Guard': (
            "Each night you protect one player, possibly yourself, or pass. The same "
            "player may not be protected two nights running. Protection stops the "
            "Wolves' kill but not the Witch's poison, and if the Witch saves the "
            "player you protected that player dies anyway - so think before "
            "doubling up on someone."
        ),
        'Hunter': (
            "You have no night action. When you are voted out, or killed by the "
            "Wolves, you may shoot one living player as you die. If you are "
            "poisoned you cannot shoot. Your threat is public, so how you vote and "
            "how openly you claim your role both matter."
        ),
        'Civilian': (
            "You have no night action and no secret knowledge. You win by reading "
            "the Wolves out of what everyone shares: who died, who voted for whom, "
            "who runs for sheriff, and who changes their story."
        ),
    },
    'zh': {
        'Wolf': "你会得知包括狼王在内的全部狼队友。每晚狼队按座位顺序商议：每人提议一名存活玩家，被提议最多者死亡——票数并列则无人死亡，所以提议分散会浪费这一晚；后发言的狼人能看到队友此前的提议。击杀好人、让自己不被怀疑就是获胜方式。发言时可以伪装成其他身份，但不要与狼队友公开矛盾。你也可以在自己发言时自爆：当天立即结束、你会出局，只有在能阻止狼队被投票放逐时才值得这么做。",
        'Wolf King': "你与狼队一同行动，并知道所有狼队友。若你被投票放逐，可以在出局时开枪带走一名存活玩家，用来换掉一个神职；但夜里被刀死时不能开枪。",
        'Prophet': "每晚你可以指定一名自己之外的存活玩家，得知其阵营；结果会累积在你的私人记录中。狼王会被验成狼人。你与好人阵营一同获胜。过早跳身份容易在夜里被杀，一直沉默又会让狼人掌控投票。",
        'Witch': "全局只有一瓶毒药和一瓶解药。你会看到今晚被狼杀的玩家：毒药可以对任意非己存活玩家使用，解药只能救今晚被狼杀的玩家（第二晚起可以自救）。毒药优先：被毒死的玩家无法被救回；若你救的人同时被守卫守护，他仍然会死。",
        'Guard': "每晚可以守护一名玩家（包括自己），也可以放弃；同一名玩家不能连续两晚被守护。守护只能挡狼刀，挡不住女巫的毒药；如果你守护的人又被女巫救下，他依然会死——所以同时保护同一个人前要谨慎。",
        'Hunter': "你没有夜间行动。被投票放逐或被狼人刀死时，可以开枪带走一名存活玩家；被毒死则不能开枪。你的威胁是公开的，所以如何投票、是否跳身份都很关键。",
        'Civilian': "你没有夜间行动，也没有秘密信息。你只能从所有人都能看到的信息里找出狼人：谁死了、谁投了谁、谁上了警、谁的发言前后矛盾。",
    },
}


CONTRACT: dict[str, str] = {'en': '''\
Reply with ONE JSON object and nothing else - no prose, no markdown fence:
{"choice": 3, "speech": "Player 3 has been quiet all day.", "reasoning": "..."}

"choice"     integer player id taken from the legal list, or null to pass when
             passing is allowed. Required for kill/save/query/vote decisions; an
             id that is not on the list is rejected, so never invent one.
"speech"     the one public sentence to announce. Only used when you are asked
             for a statement; leave it as "" otherwise - a vote or a night action
             says nothing, so analysis belongs in "reasoning".
"reasoning"  one or two private sentences: why this move, who you suspect next.
             Never shown to the other players.
"yes"        only for a yes/no question (running for sheriff, self-explosion,
             handing the badge on, choosing the speaking order): true or false.
             Omit it otherwise.
Answer with JSON only. If you cannot decide, still answer with JSON and choose
the pass option when it is legal.''',
    'zh': '''\
只回复一个 JSON 对象，不要输出任何其他内容，不要使用 markdown 代码块：
{"choice": 3, "speech": "玩家3今天一直不太说话。", "reasoning": "..."}

"choice"     从合法列表中选择的玩家编号；在允许放弃时也可以用 null 表示放弃。
             击杀/救人/查验/投票类问题必须给出编号；不在列表中的编号会被判为无效，
             不要自行发挥。
"speech"     要公开发表的那一句话。只在需要发言时使用；投票、夜晚行动都不发言，
             留空字符串，分析写进 "reasoning"。
"reasoning"  一到两句私人推理：为什么这样选择、接下来怀疑谁。不会展示给其他玩家。
"yes"        仅用于是非题（是否上警、是否自爆、是否移交警徽、发言顺序）：
             用 true 或 false 回答；其他问题不要包含这个字段。
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
        'sheriff': '- Sheriff (their vote counts 1.5): P{sheriff}',
        'sheriff_none': '- Sheriff: none this game',
        'sheriff_broken': '- The sheriff badge has been discarded',
        'candidates': '- Candidates for sheriff: {ids}',
        'candidate_said': '- Candidate statements: {said}',
        'exploded': '- Wolves who have self-exploded so far: {count}',
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
        'legal_yesno': '"yes": true  or  "yes": false',
        'corrective': ('Your previous reply was rejected: {error}\n'
                       'Previous reply was: {previous}\n'
                       'Answer again with a single valid JSON object.'),
        'kind_kill': 'kill', 'kind_pack_kill': 'pack proposal',
        'kind_save': 'save', 'kind_query': 'query',
        'kind_guard': 'guard', 'kind_vote': 'vote', 'kind_speech': 'speech',
        'kind_shot': 'shoot', 'kind_badge': 'badge', 'kind_badge_target': 'badge hand-over',
        'kind_police': 'run for sheriff',
        'kind_expose': 'self-explosion', 'kind_speech_order': 'speaking order (yes = descending seat order)',
        'kind_generic': 'action',
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
        'sheriff': '- 警长：玩家{sheriff}（其票计 1.5 票）',
        'sheriff_none': '- 本局暂无警长',
        'sheriff_broken': '- 警徽已被撕毁',
        'candidates': '- 参与竞选警长的玩家：{ids}',
        'candidate_said': '- 竞选发言：{said}',
        'exploded': '- 已自爆的狼人数量：{count}',
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
        'legal_yesno': '"yes": true 或 "yes": false',
        'corrective': ('你上一次的回复被拒绝：{error}\n'
                       '上一次的回复是：{previous}\n'
                       '请重新只输出一个合法的 JSON 对象。'),
        'kind_kill': '击杀', 'kind_pack_kill': '狼队商议', 'kind_save': '救人', 'kind_query': '查验',
        'kind_guard': '守护', 'kind_vote': '投票', 'kind_speech': '发言',
        'kind_shot': '开枪', 'kind_badge': '警徽', 'kind_badge_target': '警徽移交',
        'kind_police': '竞选警长',
        'kind_expose': '自爆', 'kind_speech_order': '发言顺序',
        'kind_generic': '行动',
        'q_pack_kill': '狼队今晚要击杀哪名玩家？（被提议最多者死亡；票数并列则无人死亡）',
        'q_save': '你要救下哪名玩家？',
        'q_query': '你要查验哪名玩家？',
        'q_guard': '你今晚要守护哪名玩家？（可以守护自己，也可以放弃）',
        'q_vote': '你要投票给谁？',
        'q_speech': '请发表你的公开陈述（一句话）。',
        'q_kill': '你要毒杀哪名玩家？（全局只有一瓶毒药）',
        'q_save': '你要用解药救下哪名玩家？（只能救今晚被狼杀的人）',
        'q_query': '你要查验哪名玩家？（会得知其阵营）',
        'q_shot': '你可以开枪带走一名存活玩家。选择谁？',
        'q_badge': '你要把警徽移交给其他玩家吗？',
        'q_badge_target': '你要把警徽移交给哪名存活玩家？',
        'q_police': '你要竞选警长吗？',
        'q_expose': '你要自爆狼人身份吗？自爆会立即出局，并跳过今天的投票。',
        'q_speech_order': '你希望从你开始按哪种顺序发言？（是=座位号从大到小，否=从小到大）',
        'q_generic': '请按上面的规则做出你的选择。',
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

    if request.kind in YES_NO_KINDS:
        legal = text['legal_yesno']
    elif request.allowed is None:
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


_CLOSER_OF = {'}': '{', ']': '['}
_PYTHON_ONLY = frozenset(')\uff09')


def _repair_json(text: str, close: bool = False) -> str:
    '''
    Fix the slips that make json.loads throw away an otherwise usable reply.

    Seen in real games: a raw control character inside a string (a literal
    newline in a multi-line "reasoning"), a trailing comma, and a stray closing
    bracket - a model once wrote
    ``{"choice": 3, "reasoning": "...。" )}`` and lost the whole turn to it. A
    ``)`` is never valid outside a JSON string, and a ``}``/``]`` that does not
    close the innermost open bracket closes nothing, so both are dropped.

    `close=True` additionally terminates what a reply cut short mid-object (a
    small max_tokens does that): the open string is closed and the open brackets
    are closed in order, so the fields that did arrive can still be read.
    '''
    chars: list[str] = []
    stack: list[str] = []
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
            continue
        if char in '{[':
            stack.append(char)
            chars.append(char)
            continue
        if char in _CLOSER_OF:
            if stack and stack[-1] == _CLOSER_OF[char]:
                stack.pop()
                chars.append(char)
            continue                      # an unmatched closer closes nothing
        if char in _PYTHON_ONLY:
            continue                      # a Python slice/paren leaked into JSON
        chars.append(char)
    if close:
        if in_string:
            # an escape still pending would swallow the quote we add, so give it
            # one of its own first
            chars.extend('""' if escaped else '"')
        while stack:
            chars.append('}' if stack.pop() == '{' else ']')
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
        for form in (candidate, _repair_json(candidate), _repair_json(candidate, close=True)):
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


_YES_WORDS = {'yes', 'y', 'true', '1', '是', '对', '好', '同意', '上警', '自爆'}
_NO_WORDS = {'no', 'n', 'false', '0', '否', '不', '不要', '不同意'}


def _coerce_yes(payload: dict[str, Any]) -> bool | None:
    '''
    Read a yes/no answer however the model phrased it.

    `yes` is the documented field; a model that reuses `choice` or `answer`
    with true/false or 是/否 is still understood rather than punished.
    '''
    for field in ('yes', 'answer', 'choice'):
        if field not in payload:
            continue
        value = payload[field]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _YES_WORDS:
            return True
        if text in _NO_WORDS:
            return False
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

    if request.kind in YES_NO_KINDS:
        yes = _coerce_yes(payload)
        if yes is None:
            raise DecodeError('this is a yes/no question: reply {"yes": true} or {"yes": false}')
        return Decision(request.kind, None, speech.strip(), reasoning.strip(), raw, yes=yes)

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
# '' means "whatever the provider defaults to"; 'off' disables thinking.
REASONING_EFFORTS = frozenset({'', 'off', 'low', 'high', 'max'})

# A ceiling for the automatic budget growth, so a runaway reply cannot ask for
# an absurd amount. Well under DeepSeek's 384K, far above any real answer.
MAX_OUTPUT_TOKENS = 32768

DEFAULT_BASE_URLS = {
    'deepseek': 'https://api.deepseek.com/v1',
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com',
    'compatible': '',        # any other OpenAI-shaped gateway: supply base_url
}
DEFAULT_MODELS = {
    # The default stays the long-standing chat model: a seat that wants DeepSeek
    # V4.1 (deepseek-flash) or deepseek-v4-pro asks for it in models.json.
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
    # A thinking model spends tokens before it answers, so this has to leave room
    # for the chain of thought as well as the JSON.
    max_tokens: int = 1024
    timeout: float = 120.0
    json_mode: bool = False       # ask for response_format=json_object when supported
    # How hard a thinking model should think: '' keeps the provider default,
    # 'off' disables thinking, 'low'/'high'/'max' map to reasoning_effort.
    reasoning_effort: str = ''

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
        self.reasoning_effort = (self.reasoning_effort or '').strip().lower()
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise LLMError(
                f'unknown reasoning_effort {self.reasoning_effort!r}; expected one of '
                + ', '.join(sorted(value for value in REASONING_EFFORTS if value))
            )

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
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1024')),
            json_mode=os.getenv('LLM_JSON_MODE', '').lower() in {'1', 'true', 'yes'},
            reasoning_effort=os.getenv('LLM_REASONING_EFFORT', ''),
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
            'reasoning_effort': self.reasoning_effort,
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
        # A thinking model's chain of thought, kept for the caller to show: the
        # JSON answer stays the only thing the engine sees.
        self.last_reasoning = ''

    def _thinking_payload(self) -> dict[str, Any]:
        '''
        Ask for a specific amount of thinking, on the providers that document it.

        DeepSeek takes `reasoning_effort` (low/high/max) and disables thinking with
        {"thinking": {"type": "disabled"}}. Other OpenAI-shaped gateways reject
        unknown fields, so nothing is sent unless the effort was asked for and the
        provider kind is known to understand it.
        '''
        effort = self.config.reasoning_effort
        if not effort or self.config.kind != 'deepseek':
            return {}
        if effort == 'off':
            return {'thinking': {'type': 'disabled'}}
        return {'reasoning_effort': effort}

    def complete(self, system: str, user: str) -> str:
        self.last_reasoning = ''
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
        payload.update(self._thinking_payload())

        data = _post_json(
            f'{self.config.base_url}/chat/completions',
            payload,
            {'Authorization': f'Bearer {self.config.api_key}'},
            self.config.timeout,
        )
        try:
            choice = data['choices'][0]
            message = choice['message']
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f'unexpected completion shape: {json.dumps(data)[:300]}') from error
        # DeepSeek returns its chain of thought as reasoning_content; some gateways
        # use "reasoning", some a list of parts.
        reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
        if isinstance(reasoning, list):
            reasoning = ' '.join(str(part) for part in reasoning)
        self.last_reasoning = str(reasoning).strip()
        content = message.get('content') or ''
        if not content.strip() and choice.get('finish_reason') == 'length':
            raise self._budget_error(len(self.last_reasoning))
        return content

    def budget_hint(self, minimum: int = 0) -> int:
        '''
        A larger budget that leaves room for the answer after the thought.

        A thought that already outran the budget will not fit in double of it, so
        grow by 4x (2048 -> 8192), with a floor for small budgets and a cap that
        still leaves the reply somewhere to land.
        '''
        return min(MAX_OUTPUT_TOKENS, max(minimum, self.config.max_tokens * 4, 4096))

    def _budget_error(self, reasoning_chars: int) -> BudgetExhausted:
        effort = self.config.reasoning_effort or 'default'
        return BudgetExhausted(
            f'ran out of {self.config.max_tokens} tokens while still thinking '
            f'(finish_reason=length, {reasoning_chars} chars of reasoning, effort={effort}); '
            f'raise max_tokens or lower reasoning_effort')

    def widen_budget(self, minimum: int = 0) -> int:
        '''Grow the output budget after a truncated thought; returns the new value.'''
        self.config.max_tokens = self.budget_hint(minimum)
        return self.config.max_tokens


class AnthropicProvider:
    name = 'anthropic'

    def __init__(self, config: ProviderConfig) -> None:
        if not config.api_key:
            raise LLMError(
                'no API key configured - set ANTHROPIC_API_KEY (or LLM_API_KEY)'
            )
        self.config = config
        self.last_reasoning = ''

    def complete(self, system: str, user: str) -> str:
        self.last_reasoning = ''
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
            blocks = data['content']
            texts = [b['text'] for b in blocks if b.get('type') == 'text']
            self.last_reasoning = ' '.join(
                str(b.get('thinking') or '') for b in blocks
                if b.get('type') in ('thinking', 'redacted_thinking')).strip()
        except (KeyError, TypeError) as error:
            raise LLMError(f'unexpected completion shape: {json.dumps(data)[:300]}') from error
        text = ''.join(texts)
        if not text.strip() and data.get('stop_reason') == 'max_tokens':
            raise BudgetExhausted(
                f'ran out of {self.config.max_tokens} tokens before answering '
                f'(stop_reason=max_tokens); raise max_tokens')
        return text

    def widen_budget(self, minimum: int = 0) -> int:
        self.config.max_tokens = min(MAX_OUTPUT_TOKENS,
                                     max(minimum, self.config.max_tokens * 4, 4096))
        return self.config.max_tokens


# -- per-seat models -------------------------------------------------------
# A game can put a different model on every seat: one seat on DeepSeek, another
# on GPT, another on Kimi. models.json names a few profiles and says which seat
# uses which; nothing here is required, and with no file every seat keeps using
# the single model configured in the console.
MODEL_FIELDS = frozenset({
    'kind', 'model', 'base_url', 'api_key', 'api_key_env', 'key_file',
    'temperature', 'max_tokens', 'timeout', 'json_mode', 'reasoning_effort',
})


@dataclass(frozen=True)
class ModelSpec:
    '''One named model from models.json, resolvable against a fallback config.'''
    name: str
    fields: dict[str, Any]

    def key_source(self, kind: str) -> str:
        '''
        Where this profile's key would come from, for display only.

        Never the key itself: the console must be able to say "a key is set, from
        models.json" without the value ever reaching the browser.
        '''
        if str(self.fields.get('api_key') or '').strip():
            return 'models.json'
        if str(self.fields.get('key_file') or '').strip():
            return f"file {self.fields['key_file']}"
        if str(self.fields.get('api_key_env') or '').strip():
            return f"env {self.fields['api_key_env']}"
        if kind == 'compatible':
            return 'not set - name api_key_env or key_file'
        return f"env {KEY_ENV.get(kind, 'LLM_API_KEY')}"

    def _resolve_key(self, kind: str, fallback: ProviderConfig | None) -> str:
        '''api_key, else key_file, else api_key_env, else the usual environment.'''
        fields = self.fields
        key = str(fields.get('api_key') or '').strip()
        if not key:
            key_file = str(fields.get('key_file') or '').strip()
            if key_file:
                try:
                    key = Path(key_file).expanduser().read_text(encoding='utf-8').strip()
                except OSError:
                    key = ''
        if not key:
            env_name = str(fields.get('api_key_env') or '').strip()
            key = os.getenv(env_name, '').strip() if env_name else ''
        if not key and fallback is not None and fallback.kind == kind:
            # the same provider, so the one key pasted in ds_key.txt serves both
            key = fallback.api_key
        if not key and kind != 'compatible':
            # only this provider's own variable: LLM_API_KEY is generic and may
            # hold a different vendor's key, which must not be sent elsewhere
            env_name = KEY_ENV.get(kind, '')
            key = os.getenv(env_name, '').strip() if env_name else ''
        return key

    def build(self, fallback: ProviderConfig | None) -> ProviderConfig:
        '''
        Turn the profile into a live config.

        Anything the profile leaves out is inherited from the fallback config
        when it is the same provider kind - that is how a seat can pin
        'deepseek-flash' while the key still comes from the one place it was
        pasted. A profile for another kind resolves its own key.
        '''
        fields = self.fields
        kind = str(fields.get('kind') or (fallback.kind if fallback else 'deepseek')).strip().lower()
        same_kind = fallback is not None and fallback.kind == kind
        key = self._resolve_key(kind, fallback)

        def pick(field: str, default: Any) -> Any:
            if field in fields:
                return fields[field]
            if same_kind:
                return getattr(fallback, field)
            return default

        return ProviderConfig(
            kind=kind,
            base_url=str(pick('base_url', '') or ''),
            model=str(pick('model', '') or ''),
            api_key=key,
            temperature=float(pick('temperature', 0.8)),
            max_tokens=int(pick('max_tokens', 1024)),
            timeout=float(pick('timeout', 120.0)),
            json_mode=bool(pick('json_mode', False)),
            reasoning_effort=str(pick('reasoning_effort', '') or ''),
        )


@dataclass
class ModelRegistry:
    '''Named profiles and the seat -> profile assignment, with a default.'''
    models: dict[str, ModelSpec]
    seats: dict[int, str]
    default: str = ''
    source: str = ''

    def profile_for(self, seat: int) -> str:
        '''The profile name for a seat; '' means "whatever the console holds".'''
        return self.seats.get(seat, self.default)

    def config_for(self, seat: int, fallback: ProviderConfig) -> ProviderConfig:
        name = self.profile_for(seat)
        spec = self.models.get(name)
        return spec.build(fallback) if spec is not None else fallback

    def describe(self, fallback: ProviderConfig | None = None) -> dict[str, Any]:
        '''Browser-safe: which model each seat plays, and whether it has a key.'''
        models: dict[str, dict[str, Any]] = {}
        for name, spec in self.models.items():
            try:
                config = spec.build(fallback)
                models[name] = {'kind': config.kind, 'model': config.model,
                                'base_url': config.base_url,
                                'reasoning_effort': config.reasoning_effort,
                                'max_tokens': config.max_tokens,
                                'has_key': bool(config.api_key),
                                'key_source': spec.key_source(config.kind)}
            except LLMError as error:
                models[name] = {'error': str(error)}
        resolved: dict[str, dict[str, Any]] = {}
        for seat in range(12):
            name = self.profile_for(seat)
            spec = self.models.get(name)
            if spec is None:
                # no profile: the seat uses whatever the environment configured
                resolved[str(seat)] = {
                    'profile': '',
                    'kind': fallback.kind if fallback else '',
                    'model': fallback.model if fallback else '',
                }
                continue
            try:
                config = spec.build(fallback)
                resolved[str(seat)] = {'profile': name, 'kind': config.kind,
                                       'model': config.model}
            except LLMError as error:
                resolved[str(seat)] = {'profile': name, 'error': str(error)}
        return {'source': self.source, 'default': self.default,
                'models': models, 'seats': resolved}


def parse_model_config(data: Any, source: str = 'models.json') -> ModelRegistry:
    '''Validate the JSON shape; every mistake names the field that caused it.'''
    if not isinstance(data, dict):
        raise LLMError(f'{source}: the top level must be an object')
    unknown = set(data) - {'version', 'default', 'models', 'seats', 'note'}
    if unknown:
        raise LLMError(f'{source}: unknown top-level key(s): ' + ', '.join(sorted(unknown)))
    raw_models = data.get('models')
    if not isinstance(raw_models, dict) or not raw_models:
        raise LLMError(f'{source}: "models" must be a non-empty object of profiles')

    models: dict[str, ModelSpec] = {}
    for name, profile in raw_models.items():
        if not isinstance(profile, dict):
            raise LLMError(f'{source}: model {name!r} must be an object')
        documented = {k for k in profile if str(k).startswith('_') or k == 'note'}
        bad = set(profile) - MODEL_FIELDS - documented
        if bad:
            raise LLMError(f'{source}: model {name!r} has unknown field(s): '
                           + ', '.join(sorted(bad)))
        # "_comment"/"note" are documentation inside a profile, not settings
        fields = {k: v for k, v in profile.items()
                  if not str(k).startswith('_') and k != 'note'}
        key_file = str(fields.get('key_file') or '').strip()
        if key_file and not Path(key_file).expanduser().is_absolute():
            # a path in models.json is relative to that file, not to the cwd
            fields['key_file'] = str((Path(source).resolve().parent / key_file))
        models[str(name)] = ModelSpec(str(name), fields)

    default = str(data.get('default') or '')
    if default and default not in models:
        raise LLMError(f'{source}: default {default!r} is not one of the models '
                       + ', '.join(sorted(models)))

    seats: dict[int, str] = {}
    raw_seats = data.get('seats') or {}
    if not isinstance(raw_seats, dict):
        raise LLMError(f'{source}: "seats" must map seat numbers to model names')
    for raw_seat, name in raw_seats.items():
        try:
            seat = int(raw_seat)
        except (TypeError, ValueError):
            raise LLMError(f'{source}: seat {raw_seat!r} is not a number') from None
        if not 0 <= seat < 12:
            raise LLMError(f'{source}: seat {seat} is outside the table (0-11)')
        if name not in models:
            raise LLMError(f'{source}: seat {seat} names unknown model {name!r}')
        seats[seat] = str(name)

    return ModelRegistry(models=models, seats=seats, default=default, source=source)


def load_model_config(path: str | Path) -> ModelRegistry:
    '''Read models.json. A broken file is an error, not a silent fallback.'''
    path = Path(path)
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as error:
        raise LLMError(f'cannot read {path}: {error}') from error
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMError(f'{path} is not valid JSON ({error.msg} at line {error.lineno})') from error
    return parse_model_config(data, source=str(path))


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
            except BudgetExhausted as error:
                # Retrying the same request would truncate again, so grow the
                # budget first. The thought itself is worth keeping: it is what
                # the console shows while the model works.
                failure = str(error)
                widen = getattr(self.provider, 'widen_budget', None)
                if widen is not None and attempt < self.attempts:
                    budget = widen()
                    self._emit(kind='llm-error', viewer=observation.viewer,
                               attempt=attempt, text=f'{failure} -> retrying with max_tokens={budget}')
                else:
                    self._emit(kind='llm-error', viewer=observation.viewer,
                               attempt=attempt, text=failure)
                continue
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
            if not decision.reasoning:
                # a thinking model explains itself outside the JSON; keeping that
                # text means the console can show why it moved
                decision.reasoning = str(getattr(self.provider, 'last_reasoning', '') or '')
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
        if request.kind in YES_NO_KINDS:
            # "no" is the harmless answer: it does not explode and it tears the
            # badge rather than handing it to a random player.
            return Decision(request.kind, note=note, yes=False)
        pick = request.allowed[0] if request.required and request.allowed else None
        return Decision(request.kind, pick, note=note)
