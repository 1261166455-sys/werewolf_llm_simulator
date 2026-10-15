'''
Language handling for the console.

Three layers are translated:

  * the description prompt sent to the model   - llm_interface.py
  * the text engine.py prints                    - translated here, on the way out
  * the console page                           - static/index.html

engine.py deliberately keeps its English strings. The bridge translates them as
they leave, which means the language can be switched at runtime, and
llm_interface.classify() still matches the originals it depends on. A string
with no entry here passes through unchanged rather than being dropped, so a new
message in engine.py shows up in English instead of disappearing.
'''
from __future__ import annotations

import re
from typing import Any

LANGUAGES = {'zh': '中文', 'en': 'English'}
DEFAULT = 'zh'

_language = DEFAULT


def get_language() -> str:
    return _language


def set_language(code: str) -> str:
    '''Switch language. Returns the language actually in force.'''
    global _language
    if code in LANGUAGES:
        _language = code
    return _language


def is_supported(code: str) -> bool:
    return code in LANGUAGES


# -- text engine.py emits ----------------------------------------------------
_PLAYER = re.compile(r'Player (\d+)')

# Applied in order, so the specific sentences run before the generic rewrite of
# "Player n" that catches anything left over.
_ENGINE: dict[str, list[tuple[re.Pattern[str], str]]] = {
    'en': [],
    'zh': [
        (re.compile(r"^Welcome to the game!\s*$"), '游戏开始！'),
        (re.compile(r'^Night (\d+) approach\s*$'), r'第 \1 个夜晚来临'),
        (re.compile(r'^Tonight nobody died\.\s*$'), '今晚无人死亡。'),
        (re.compile(r'^Tonight (.+) died\.\s*$'), r'今晚 \1 死亡。'),
        (re.compile(r'^Since nobody died tonight, you cannot save anyone\s*$'),
         '今晚无人死亡，无法使用解药'),
        (re.compile(r'^Night ends and day approches\s*$'), '夜晚结束，白天来临'),
        (re.compile(r'^Death\(s\) at night (\d+): (.+)$'), r'第 \1 夜死亡：\2'),
        (re.compile(r'^No death at night (\d+)\s*$'), r'第 \1 夜无人死亡'),
        (re.compile(r'^Messages summary:\s*$'), '发言汇总：'),
        (re.compile(r'^Player (\d+): (.*)$'), r'玩家\1：\2'),
        (re.compile(r'^Start voting\s*$'), '开始投票'),
        (re.compile(r'^Player (\d+) out \((\d+) votes, by (.+)\)$'),
         r'玩家\1 被投出（\2 票，来自 \3）'),
        (re.compile(r'^No one was out\. Since candidates (.+) have same votes\s*$'),
         r'无人出局：\1 得票相同'),
        (re.compile(r'^All wolves dead, civilians win!\s*$'),
         '所有狼人已死亡，好人阵营获胜！'),
        (re.compile(r'^The prophet and witch were dead\s*$'), '预言家与女巫均已死亡'),
        (re.compile(r'^All civilians are dead\s*$'), '所有平民均已死亡'),
        (re.compile(r'^Wolves win!\s*$'), '狼人阵营获胜！'),
        (re.compile(r'^Game Tie! \(more than (\d+) days passed\)\s*$'),
         r'平局！（已超过 \1 天）'),
        (re.compile(r'^Game continue\. Prepare for the next night\s*$'),
         '游戏继续，请准备下一个夜晚'),
        (re.compile(r"^Good game! Here's a replay for last game:\s*$"),
         '游戏结束！以下是上一局的回放：'),
        (re.compile(r'^Empty or invalid format, please re-input\s*$'),
         '输入为空或格式错误，请重新输入'),
        (re.compile(r'^Invalid target chosen, please re-input\s*$'),
         '目标不合法，请重新输入'),
        # the prompts themselves
        (re.compile(r'^\[Wolf\] Please input an player ID to kill'
                    r' \(leave blank to skip\):\s*$'),
         '[狼人] 请输入要击杀的玩家编号（留空表示跳过）：'),
        (re.compile(r"^\[Prophet\] Input a player's ID to query its state:\s*$"),
         '[预言家] 请输入要查验的玩家编号：'),
        (re.compile(r"^\[Witch\] Input a player's ID to kill:\s*$"),
         '[女巫] 请输入要毒杀的玩家编号：'),
        (re.compile(r"^\[Witch\] Input a player's ID to save"
                    r" \(Please select one who was killed tonight\):\s*$"),
         '[女巫] 请输入要救的玩家编号（请选今晚被杀的玩家）：'),
        (re.compile(r'^\[Vote\] Vote whom\?\s*$'), '[投票] 你要投票给谁？'),
        (re.compile(r'^\[Addresser\]\s*$'), '[发言]'),
    ],
}


def _zh_players(text: str) -> str:
    '''Reword any surviving "Player n" references and join lists the Chinese way.'''
    text = _PLAYER.sub(lambda m: f'玩家{m.group(1)}', text)
    return re.sub(r',\s*玩家', '、玩家', text)


def engine_text(text: str, language: str | None = None) -> str:
    '''Translate one message produced by engine.py (or return it unchanged).'''
    lang = language or _language
    if not text or lang == 'en':
        return text
    for pattern, replacement in _ENGINE.get(lang, ()):
        if pattern.match(text):
            # the replacement can itself still contain "Player 1, Player 2"
            # (death lists, vote lists), so run the seat rewrite on the result
            return _zh_players(pattern.sub(replacement, text)) if lang == 'zh' else pattern.sub(replacement, text)
    return _zh_players(text) if lang == 'zh' else text


# -- names -----------------------------------------------------------------
JOB_NAMES = {
    'zh': {'Wolf': '狼人', 'Prophet': '预言家', 'Witch': '女巫',
           'Civilian': '平民', 'Not_Assigned': '未分配'},
    'en': {'Wolf': 'Wolf', 'Prophet': 'Prophet', 'Witch': 'Witch',
           'Civilian': 'Civilian', 'Not_Assigned': 'Unknown'},
}

PLAYER_STATES = {
    'zh': {'Alive': '存活', 'Dead': '死亡', 'Exit': '出局'},
    'en': {'Alive': 'Alive', 'Dead': 'Dead', 'Exit': 'Exit'},
}

STAGES = {
    'zh': {'Initializing': '初始化', 'Finished_Initializing': '初始化完成',
           'Night': '夜晚', 'Day': '白天', 'End_Civilian_Win': '好人阵营获胜',
           'End_Wolf_Win': '狼人阵营获胜', 'End_Tie': '平局'},
    'en': {},
}

CAMP_NAMES = {'zh': {'Civilians': '好人阵营', 'Wolves': '狼人阵营', 'Tie': '平局'}, 'en': {}}


def name(table: dict[str, dict[str, str]], key: str, language: str | None = None) -> str:
    lang = language or _language
    return table.get(lang, {}).get(key) or key


def player_ids(values: Any, language: str | None = None) -> str:
    '''Render a sequence of seat numbers for a prompt, e.g. "玩家1、玩家3".'''
    lang = language or _language
    values = tuple(values)
    if not values:
        return '无人' if lang == 'zh' else 'nobody'
    if lang == 'zh':
        return '、'.join(f'玩家{v}' for v in values)
    return ', '.join(f'P{v}' for v in values)


# -- strings the console itself emits -------------------------------------
MESSAGES: dict[str, dict[str, str]] = {
    'busy': {'en': 'a game is already in progress',
             'zh': '已有一局游戏正在进行'},
    'nothing_to_replay': {'en': 'nothing recorded yet - play a game first',
                          'zh': '暂无可回放内容，请先进行一局游戏'},
    'stray_answer': {'en': 'answer ignored: nothing was waiting for input',
                     'zh': '忽略该输入：当前没有等待输入的提示'},
    'unknown_message': {'en': 'unknown message: {what}',
                        'zh': '未知消息：{what}'},
    'contacting': {'en': 'contacting the model…', 'zh': '正在连接模型…'},
    'model_ready': {'en': 'model ready - answered a test vote ({picked})',
                    'zh': '模型就绪——已正确回答一次测试投票（{picked}）'},
    'conn_failed': {'en': 'connection test failed: {error}',
                    'zh': '连接测试失败：{error}'},
    'decode_failed': {
        'en': 'reached the model, but its reply did not decode ({error}). Raw reply: {raw}',
        'zh': '已连接模型，但回复无法解析（{error}）。原始回复：{raw}'},
    'bad_settings': {'en': 'invalid LLM settings: {error}',
                     'zh': '模型设置无效：{error}'},
    'choose_pass': {'en': 'pass', 'zh': '放弃'},
    'choose_choice': {'en': 'choice={choice}', 'zh': '选择 {choice} 号'},
    'ended_tie': {'en': 'ended in a tie', 'zh': '以平局结束'},
    'win_message': {'en': '{camp} win', 'zh': '{camp}获胜'},
    'replay_ended': {'en': 'replay ended', 'zh': '回放结束'},
    'fallback_speech': {'en': 'No comment.', 'zh': '（无话可说）'},
    'fallback_note': {'en': 'fallback: {error}', 'zh': '已回退：{error}'},
    'probe_speech': {'en': 'Readiness check - this is not a real game.',
                     'zh': '连接自检——这不是真实对局。'},
    'language_set': {'en': 'language set to {name}', 'zh': '界面语言已切换为{name}'},
    # launcher banner
    'banner_title': {'en': 'Werewolf operator console', 'zh': '狼人杀操作台'},
    'banner_seats': {'en': '  seats     : {model} model / {human} human',
                     'zh': '  座位      : {model} 名模型 / {human} 名真人'},
    'banner_model': {'en': '  model     : {kind} · {model}',
                     'zh': '  模型      : {kind} · {model}'},
    'banner_endpoint': {'en': '  endpoint  : {url}', 'zh': '  接口地址  : {url}'},
    'banner_key_loaded': {'en': '  api key   : key loaded from {file}',
                          'zh': '  密钥      : 已从 {file} 读取密钥'},
    'banner_key_missing': {'en': '  api key   : NO KEY', 'zh': '  密钥      : 未找到密钥'},
    'banner_page': {'en': '  page      : {url}', 'zh': '  页面      : {url}'},
    'key_missing_hint': {
        'en': '! no key found in {file} - human seats still work, and a key can be pasted '
              'into the console under "Model & seats".',
        'zh': '！未在 {file} 中找到密钥——真人座位仍可使用，也可以在页面“模型与座位”中粘贴密钥。'},
    'port_busy': {'en': '! port {port} is busy, using {used} instead',
                  'zh': '！端口 {port} 被占用，改用 {used}'},
    # export / replay files
    'exported': {'en': 'exported {steps} steps to {file}',
                 'zh': '已导出 {steps} 步到 {file}'},
    'nothing_to_export': {'en': 'nothing to export yet - play a game first',
                          'zh': '暂无可导出的内容，请先进行一局游戏'},
    'replay_missing': {'en': 'no such export: {file}', 'zh': '找不到导出文件：{file}'},
    'replay_bad': {'en': 'cannot read that export ({error})',
                   'zh': '无法读取该导出文件（{error}）'},
    'replaying_file': {'en': 'replaying {file} ({steps} steps)',
                       'zh': '正在回放 {file}（{steps} 步）'},
    'unknown_language': {'en': 'unsupported language: {code}',
                         'zh': '不支持的语言：{code}'},
    # private facts handed to a model
    'fact_divination': {'en': 'Your own divination: Player {target} is {verdict}.',
                        'zh': '你的查验结果：玩家{target} 是{verdict}。'},
    'verdict_wolf': {'en': 'a Wolf', 'zh': '狼人'},
    'verdict_clean': {'en': 'not a Wolf', 'zh': '好人'},
    'fact_fellow_wolf': {'en': 'Your fellow Wolf is {ids}.',
                         'zh': '你的狼队友是 {ids}。'},
    'fact_pack_named': {'en': 'Tonight the pack has named {ids} to die.',
                        'zh': '今晚狼队已指定 {ids} 出局。'},
    'fact_wolves_killed': {'en': 'Tonight the Wolves killed {ids}.',
                           'zh': '今晚狼人击杀了 {ids}。'},
    'fact_you_killed': {'en': 'You chose to kill Player {player} tonight.',
                        'zh': '你今晚选择毒杀玩家{player}。'},
    'fact_you_saved': {'en': 'You chose to save Player {player} tonight.',
                       'zh': '你今晚选择救下玩家{player}。'},
    # round history lines
    'history_round': {
        'en': 'Round {round}: statements [{said}]; votes [{votes}]; left play [{left}]',
        'zh': '第 {round} 轮：发言 [{said}]；投票 [{votes}]；出局 [{left}]'},
    'history_none': {'en': 'none', 'zh': '无'},
    'history_nobody': {'en': 'nobody', 'zh': '无人'},
    'history_said': {'en': 'P{player}: {text}', 'zh': '玩家{player}：{text}'},
    'history_votes': {'en': 'P{player}:{count}', 'zh': '玩家{player}:{count}'},
}


def t(key: str, language: str | None = None, **kwargs: Any) -> str:
    '''Look up a console string, formatted. Unknown keys return the key itself.'''
    lang = language or _language
    entry = MESSAGES.get(key)
    if not entry:
        return key
    template = entry.get(lang) or entry.get('en') or key
    return template.format(**kwargs) if kwargs else template
