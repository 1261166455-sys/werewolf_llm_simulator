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
from typing import Any, Callable

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

# The phase ids in GameRule.Night / GameRule.Day, as the engine announces them.
# The announcement itself is deliberately liveness-blind: it uses the same words
# whether or not the job still has a living holder.
PHASE_NAMES = {
    'zh': {'guard_do': '守卫', 'hunter_do': '猎人', 'wolf_do': '狼人', 'witch_do': '女巫',
           'prophet_do': '预言家', 'civilian_do': '平民',
           'comment': '发言', 'vote_and_exile': '投票', 'determine_game_process': '结算'},
    'en': {},
}


def _phase_zh(name: str) -> str:
    return PHASE_NAMES['zh'].get(name, name)


# Applied in order, so the specific sentences run before the generic rewrite of
# "Player n" that catches anything left over.
_ENGINE: dict[str, list[tuple[re.Pattern[str], str | Callable[[re.Match[str]], str]]]] = {
    'en': [],
    'zh': [
        (re.compile(r"^Welcome to the game!\s*$"), '游戏开始！'),
        (re.compile(r'^Night (\d+) approach\s*$'), r'第 \1 个夜晚来临'),
        # the per-phase envelope, announced identically whoever is alive
        (re.compile(r'^\[([a-z_]+)\] begins\s*$'),
         lambda m: f'[{_phase_zh(m.group(1))}] 阶段开始'),
        (re.compile(r'^\[([a-z_]+)\] ends\s*$'),
         lambda m: f'[{_phase_zh(m.group(1))}] 阶段结束'),
        (re.compile(r'^Tonight nobody died\.\s*$'), '今晚无人死亡。'),
        (re.compile(r'^Tonight (.+) died\.\s*$'), r'今晚 \1 死亡。'),
        (re.compile(r'^Potions left: (\d+) poison, (\d+) antidote\.\s*$'),
         r'剩余药水：毒药 \1 瓶，解药 \2 瓶。'),
        (re.compile(r'^Since nobody died tonight, you cannot save anyone\s*$'),
         '今晚无人死亡，无法使用解药'),
        (re.compile(r'^No player you may save tonight\.\s*$'),
         '今晚没有可以救的玩家（不能自救或解药救不活被毒死的人）。'),
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
        (re.compile(r'^All gods are dead\s*$'), '所有神职均已死亡'),
        (re.compile(r'^All civilians are dead\s*$'), '所有平民均已死亡'),
        (re.compile(r'^Wolves win!\s*$'), '狼人阵营获胜！'),
        (re.compile(r'^Game Tie! \(more than (\d+) days passed\)\s*$'),
         r'平局！（已超过 \1 天）'),
        (re.compile(r'^Game continue\. Prepare for the next night\s*$'),
         '游戏继续，请准备下一个夜晚'),
        (re.compile(r'^Good game! The game finally ends, lasting (\d+) cycles\s*$'),
         r'游戏结束，共进行 \1 轮'),
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
        # -- police election and badge ------------------------------------
        (re.compile(r'^Now we can select a police\. You can select whether you will be an candidate\s*$'),
         '现在开始竞选警长，请选择是否上警'),
        (re.compile(r'^Next is the candidites\' speech\s*$'), '接下来是竞选者发言'),
        (re.compile(r'^Speech summary\s*$'), '竞选发言汇总：'),
        (re.compile(r'^Player (\d+) want to be the police: (.*)$'), r'玩家\1 竞选警长：\2'),
        (re.compile(r'^No one choose to be the police, skipped\s*$'), '无人竞选警长，跳过竞选'),
        (re.compile(r"^Now we're going to vote a police\s*$"), '现在开始投票选警长'),
        (re.compile(r'^No police for this game, (.+) have the same vote\s*$'),
         r'本局无警长：\1 得票相同'),
        (re.compile(r'^Since only Player (\d+) choose to be the candidate, the player will be the police\s*$'),
         r'只有玩家\1 竞选，直接成为警长'),
        (re.compile(r'^The police \(Player (\d+)\) is dead\s*$'), r'警长（玩家\1）已出局'),
        (re.compile(r'^No living player can take the badge, so it is torn up\s*$'),
         '没有存活玩家可以接警徽，警徽被撕毁'),
        (re.compile(r'^The badge was not handed over, so it is torn up\s*$'),
         '未移交警徽，警徽被撕毁'),
        (re.compile(r'^The police has determined the comment order: (.+)$'),
         r'警长已确定发言顺序：\1'),
        (re.compile(r'^Since no police, the comment order fallbacks to default order\s*$'),
         '无警长，按默认顺序发言'),
        # -- day: last words and self-explosion ---------------------------
        (re.compile(r"^It's time for deaths to speak before they are out\s*$"),
         '现在由昨夜死亡的玩家发表遗言'),
        (re.compile(r'^There exists wolf who tell his identification, skip the vote and exile process, next night approaches\s*$'),
         '有狼人自爆，跳过今天的投票放逐，直接进入下一夜晚'),
        (re.compile(r'^Player (\d+) was shot by player (\d+)\s*$'), r'玩家\2 开枪带走了玩家\1'),
        (re.compile(r'^No one was shot\s*$'), '没有开枪'),
        (re.compile(r'^Please answer y or n\s*$'), '请回答 y 或 n'),
        (re.compile(r'^Unrecognised answer, treated as no\s*$'), '无法识别的回答，按「否」处理'),
        # -- the prompts themselves --------------------------------------
        (re.compile(r'^\[Option\] Choose to be the police\? \[y/n\]\s*$'),
         '[竞选] 你要竞选警长吗？[y/n]'),
        (re.compile(r'^\[Choice\] Do you want to expose your identification\? \[y/n\]\s*$'),
         '[选择] 你要自爆狼人身份吗？[y/n]'),
        (re.compile(r'^Determine the comment order, start from you, yes for clockwise, no for counterclockwise \[y/n\]\s*$'),
         '请确定发言顺序（从你开始：是=座位号从大到小，否=从小到大）[y/n]'),
        (re.compile(r'^Do you want to transmit your badge to a new player\? \[y/n\]\s*$'),
         '你要把警徽移交给其他玩家吗？[y/n]'),
        (re.compile(r'^Which player do you want to send your badge\?\s*$'),
         '你要把警徽交给哪名存活玩家？'),
        (re.compile(r'^You can shoot 1 person, input an ID:\s*$'),
         '你可以开枪带走一名存活玩家，请输入编号：'),
        (re.compile(r'^\[Guard\] Input a player\'s ID to guard \(No double guarding\):\s*$'),
         '[守卫] 请输入要守护的玩家编号（不能连续两晚守护同一人）：'),
        (re.compile(r'^\[Speecher\]\s*$'), '[竞选发言]'),
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
    # keyed by Jobs.<member>.value, which is what the observation carries
    'zh': {'Wolf': '狼人', 'Wolf King': '狼王', 'Prophet': '预言家', 'Witch': '女巫',
           'Guard': '守卫', 'Hunter': '猎人',
           'Civilian': '平民', 'Not_Assigned': '未分配'},
    'en': {'Wolf': 'Wolf', 'Wolf King': 'Wolf King', 'Prophet': 'Prophet', 'Witch': 'Witch',
           'Guard': 'Guard', 'Hunter': 'Hunter',
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
    'model_ready': {'en': '{model} ready - answered a test vote ({picked})',
                    'zh': '{model} 就绪——已正确回答一次测试投票（{picked}）'},
    'conn_failed': {'en': 'connection test failed: {error}',
                    'zh': '连接测试失败：{error}'},
    'conn_failed_model': {'en': '{model}: connection test failed - {error}',
                          'zh': '{model}：连接测试失败——{error}'},
    'models_title': {'en': 'model configuration ({source})',
                     'zh': '模型配置（{source}）'},
    'models_none': {
        'en': 'no models.json - every seat uses {kind}/{model} from the environment',
        'zh': '未找到 models.json——所有座位都使用环境中的 {kind}/{model}'},
    'models_seat': {'en': 'seat {seat}', 'zh': '座位 {seat}'},
    'models_broken': {'en': 'models.json is invalid: {error}',
                      'zh': 'models.json 无效：{error}'},
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
    'banner_seat_model': {'en': '  P{seat:<9}: {name} ({kind} · {model}) {keytext}',
                          'zh': '  座位 {seat:<6}: {name}（{kind} · {model}）{keytext}'},
    'banner_no_key': {'en': '- NO KEY', 'zh': '—— 无密钥'},
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
    'exported_text': {'en': ' (text record: {file})', 'zh': '（文本记录：{file}）'},
    # the companion .txt export (human review only)
    'text_title': {
        'en': 'WEREWOLF GAME RECORD - for human review; this file cannot be replayed',
        'zh': '狼人杀对局文本记录 —— 仅供人工复盘，本文件不能用于回放'},
    'text_exported_at': {'en': 'exported at : {when}', 'zh': '导出时间   : {when}'},
    'text_seed': {'en': 'seed        : {seed}', 'zh': '随机种子   : {seed}'},
    'text_model': {'en': 'model       : {model}', 'zh': '模型       : {model}'},
    'text_language': {'en': 'language    : {code}', 'zh': '界面语言   : {code}'},
    'text_rounds': {'en': 'progress    : {stage}, round {cycles} ({state})',
                    'zh': '进度       : {stage}，第 {cycles} 轮（{state}）'},
    'text_finished': {'en': 'finished', 'zh': '已结束'},
    'text_unfinished': {'en': 'unfinished', 'zh': '未结束'},
    'text_result': {'en': 'result      : {result}', 'zh': '结果       : {result}'},
    'text_no_result': {'en': 'no winner yet', 'zh': '暂无胜方'},
    'text_sheriff': {'en': 'sheriff     : {sheriff}', 'zh': '警长       : {sheriff}'},
    'text_absent': {'en': 'none', 'zh': '无'},
    'text_badge_state': {'en': 'badge       : {state}', 'zh': '警徽       : {state}'},
    'text_badge_whole': {'en': 'intact', 'zh': '完好'},
    'text_badge_lost': {'en': 'torn', 'zh': '已流失'},
    'text_exploded': {'en': 'self-explode: {count} wolf(s)', 'zh': '自爆       : {count} 狼'},
    'text_seats': {'en': 'seats       : {seats}', 'zh': '座位       : {seats}'},
    'text_roles_header': {'en': '--- roles and fates ---', 'zh': '—— 角色与状态 ——'},
    'text_role_row': {'en': 'P{seat}\t{job}\t{state}', 'zh': '玩家{seat}\t{job}\t{state}'},
    'text_alive': {'en': 'alive', 'zh': '存活'},
    'text_out': {'en': 'out', 'zh': '出局'},
    'text_stream_header': {'en': '--- full record ---', 'zh': '—— 对局记录 ——'},
    'text_legend': {
        'en': 'marks: [public] everyone sees it; [private] only that seat sees it',
        'zh': '标记说明：[公开] 全场可见；[私有] 仅该座位可见'},
    'text_blank': {'en': '(blank)', 'zh': '（空）'},
    'text_public': {'en': 'public', 'zh': '公开'},
    'text_private': {'en': 'private', 'zh': '私有'},
    'text_seat_acting': {'en': 'P{seat}', 'zh': '玩家{seat}'},
    'text_asks': {'en': 'is asked:', 'zh': '被询问：'},
    'text_legal': {'en': 'legal answers:', 'zh': '可选答案：'},
    'text_answers': {'en': 'answers:', 'zh': '回答：'},
    'text_decision': {'en': 'decision', 'zh': '决策'},
    'text_speech': {'en': 'statement:', 'zh': '发言：'},
    'text_thoughts': {'en': 'thinking:', 'zh': '想法：'},
    'text_reasoning': {'en': 'reasoning:', 'zh': '理由：'},
    'text_note': {'en': 'note:', 'zh': '备注：'},
    'text_thinking': {'en': 'P{seat} is thinking ({kind})', 'zh': '玩家{seat} 思考中（{kind}）'},
    'text_error': {'en': 'model error:', 'zh': '模型错误：'},
    'text_history_dump': {'en': 'engine state dump:', 'zh': '引擎状态快照：'},
    'text_footer': {'en': 'recorded steps: {steps}', 'zh': '记录步数：{steps}'},
    'controller_human': {'en': 'human', 'zh': '真人'},
    'controller_llm': {'en': 'model', 'zh': '模型'},
    'controller_replay': {'en': 'replay', 'zh': '回放'},
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
    'fact_guarded_before': {'en': 'You have already protected {ids}, so cannot protect them again tonight.',
                            'zh': '你已经守护过 {ids}，今晚不能重复守护他们。'},
    'fact_potions': {'en': 'Your potions left: {poison} poison, {antidote} antidote.',
                     'zh': '你剩余的药水：毒药 {poison} 瓶，解药 {antidote} 瓶。'},
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
