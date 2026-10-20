'''
Web operator console for the werewolf engine.

    venv/bin/python ui.py          # then open http://127.0.0.1:8000

engine.py stays untouched. Instead this module rebinds the module-level names that
engine.py already uses for its I/O, and drives the blocking state machine from a
worker thread so the browser can answer each prompt as it appears.
'''
from __future__ import annotations

import asyncio
import builtins
import json
import pprint as pprint_module
import re
import queue
import random
import threading
import os
import time
import traceback
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import i18n
import engine
import llm_interface

HOST = '127.0.0.1'
PORT = 8000
PAGE_PATH = Path(__file__).resolve().parent / 'static' / 'index.html'
REPLAY_DIR = Path(__file__).resolve().parent / 'replays'
# Optional per-seat models. Absent or empty means every seat uses the single
# model configured in the console, which is the behaviour without the file.
MODELS_PATH = Path(os.getenv('LLM_MODEL_CONFIG')
                   or Path(__file__).resolve().parent / 'models.json')
REPLAY_FORMAT = 'werewolf-replay'
# Bumped when the payload shape changes. A file is replayable only while its
# `rule_version` matches engine.GameRule.Version: a recorded answer sequence
# means nothing once the prompts or the rules they answer have changed.
REPLAY_VERSION = 2
REPLAY_DELAY = 0.3


def replay_filename(name: str = '', fallback: str = '') -> str:
    '''A filesystem-safe name, so a chosen name can never escape the directory.'''
    cleaned = re.sub(r'[^0-9A-Za-z._\u4e00-\u9fff-]+', '_', (name or '').strip()).strip('._')
    return (cleaned[:60] or fallback) if (cleaned or fallback) else fallback

WINNERS = {
    'End_Civilian_Win': 'Civilians',
    'End_Wolf_Win': 'Wolves',
    'End_Tie': 'Tie',
}

# Events the human-review .txt renders. Everything else in `Session.events` is
# bridge bookkeeping (snapshots, seat edits, the replay list).
RENDERED_EVENTS = frozenset({'log', 'private', 'prompt', 'answer', 'decision',
                             'thinking', 'llm-error', 'history', 'notice'})

# Decisions the whole table is entitled to see. The sheriff election, the badge,
# the speaking order and a dying shot are all announced publicly by the rules;
# everything else (a kill, a save, a query, a guard, a wolf deciding whether to
# explode) would name a role, so it goes to the private panel.
PUBLIC_KINDS = frozenset({
    llm_interface.DecisionKind.Speech,
    llm_interface.DecisionKind.Vote,
    llm_interface.DecisionKind.Police,
    llm_interface.DecisionKind.Badge,
    llm_interface.DecisionKind.BadgeTarget,
    llm_interface.DecisionKind.SpeechOrder,
    llm_interface.DecisionKind.Shot,
})


class ReplayExhausted(Exception):
    '''Raised when a replay needs more answers than were recorded.'''


class Session:
    '''
    One game at a time, shared by every connected browser tab.

    The engine runs in its own thread and calls wait_for_answer() for each
    prompt, which blocks that thread until a browser posts the answer.
    '''

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[WebSocket] = set()
        self.events: list[dict[str, Any]] = []
        self.answers: list[str] = []
        self.seed: int | None = None
        self.status = 'idle'
        self.message = ''
        self.thread: threading.Thread | None = None
        self.game: engine.Game | None = None
        # Set while JobAction.input_an_id is running, so the UI can be told which
        # player IDs that prompt will actually accept.
        self.structured: dict[str, Any] | None = None
        self.pending_input = False
        self._inbox: queue.Queue[str] = queue.Queue()
        self._replay_script: list[str] | None = None
        self._replay_delay = 0.0
        self._events_lock = threading.Lock()

        # Seats: 'human' answers through the browser, 'llm' answers through
        # llm_interface. Mixed games are the point - any split works.
        self.seats: dict[int, str] = {i: 'human' for i in range(engine.PLAYERS_COUNT)}
        try:
            self.provider_config = llm_interface.ProviderConfig.from_env()
            self.config_error = ''
        except llm_interface.LLMError as error:
            self.provider_config = llm_interface.ProviderConfig()
            self.config_error = str(error)
        # One controller per seat, because each seat may play a different model.
        self.controllers: dict[int, llm_interface.LLMController] = {}
        self.models: llm_interface.ModelRegistry | None = None
        self.model_config_error = ''
        if MODELS_PATH.is_file():
            try:
                self.models = llm_interface.load_model_config(MODELS_PATH)
            except llm_interface.LLMError as error:
                self.model_config_error = str(error)
        self.show_prompts = False
        # Presentational only: the models' own observations are always filtered
        # per seat, so revealing roles here never changes what a model knows.
        self.reveal_jobs = False
        # Show each model's reasoning in the transcript as it decides. Off by
        # default like the other observation switches: it never changes what a
        # model knows, only what the human watching can read.
        self.show_thinking = False
        # Off by default: the .json is the replayable record; the extra .txt is a
        # human-review copy (thinking, private notes, prose state) that cannot be
        # fed back into the engine.
        self.raw_output = False
        self.language = i18n.get_language()

        # Per-seat private notes and the observed public timeline. The engine
        # stores only the current night/day, so the bridge accumulates what each
        # seat has learned (e.g. the Prophet's results, which engine.py drops
        # every night) and who has left play in which round.
        self.memory: dict[int, list[str]] = {}
        self.exit_log: list[tuple[int, str, tuple[int, ...]]] = []
        self.exited_seen: set[int] = set()
        self._seen_players: tuple[tuple[int, str], ...] = ()
        # which export is being replayed, for the status line
        self.copy: dict[str, Any] = {}

        # Wall-clock and monotonic timing for a real game only: a replay would
        # time the replay, not the players who originally acted. `_game_started`
        # is the monotonic anchor, `_action_started` the anchor of the decision
        # currently being asked.
        self.game_started_at: str | None = None
        self.game_finished_at: str | None = None
        self._game_started: float | None = None
        self._game_finished: float | None = None
        self._action_started: float | None = None
        self._action_meta: dict[str, Any] | None = None
        self.action_times: list[dict[str, Any]] = []

    # -- engine thread side ------------------------------------------------
    def emit(self, event: dict[str, Any], record: bool = True) -> None:
        event = dict(event)
        if record:
            with self._events_lock:
                event['seq'] = len(self.events)
                self.events.append(event)
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except Exception:
                self.clients.discard(client)

    def begin_structured_input(self, key: str, required: bool, exclude) -> None:
        # The choices offered are exactly what JobAction.input_an_id will accept,
        # so they have to follow the engine's table size
        self.structured = {
            'key': key,
            'required': bool(required),
            'choices': [i for i in range(engine.PLAYERS_COUNT) if i not in exclude],
        }

    def end_structured_input(self) -> None:
        self.structured = None

    # -- timing ------------------------------------------------------------
    def _reset_timing(self) -> None:
        '''Forget the previous game's clock. Called when a game starts.'''
        self.game_started_at = None
        self.game_finished_at = None
        self._game_started = None
        self._game_finished = None
        self._action_started = None
        self._action_meta = None
        self.action_times = []

    def timing_snapshot(self) -> dict[str, Any]:
        '''
        The live clock for the console and the `snapshot` message.

        `total_seconds` freezes once the game ends; `action_seconds` is the
        decision in progress and is 0 between decisions, which is what "resets
        when a player finishes" means. `recorded` is false for idle sessions and
        replays, so the console knows to hide the clock.
        '''
        if self._game_started is None:
            return {'recorded': False, 'running': False,
                    'total_seconds': 0.0, 'action_seconds': 0.0, 'player': None}
        end = self._game_finished if self._game_finished is not None else time.monotonic()
        action = 0.0
        if self._action_started is not None:
            action = max(0.0, time.monotonic() - self._action_started)
        return {
            'recorded': True,
            'running': self.status == 'running',
            'total_seconds': round(max(0.0, end - self._game_started), 3),
            'action_seconds': round(action, 3),
            'player': (self._action_meta or {}).get('player'),
        }

    def _begin_action(self, prompt: str, player: int, meta: dict[str, Any]) -> bool:
        '''Start the clock for one decision. Returns False when not timed.'''
        if self._game_started is None or self.status != 'running':
            return False
        state = self.game.game_state if self.game is not None else None
        self._action_meta = {
            'player': player,
            'role': state.players[player].job.value if state and state.players else None,
            'decision': llm_interface.classify(str(meta.get('key') or ''), prompt).value,
            'controller': 'llm' if self.seats.get(player) == 'llm' else 'human',
        }
        self._action_started = time.monotonic()
        # Not part of the transcript: the clock is live state, and the export
        # keeps the full record under `timing` instead.
        self.emit({'kind': 'timing', **self.timing_snapshot()}, record=False)
        return True

    def _end_action(self) -> None:
        '''Record the finished decision and reset the action clock to zero.'''
        if self._action_started is None or self._action_meta is None:
            return
        seconds = max(0.0, time.monotonic() - self._action_started)
        state = self.game.game_state if self.game is not None else None
        record = {
            'index': len(self.action_times),
            **self._action_meta,
            'cycle': state.cycles if state is not None else None,
            'stage': state.stage.name if state is not None else None,
            'started_at': time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(time.time() - seconds)),
            'seconds': round(seconds, 3),
        }
        self.action_times.append(record)
        self._action_started = None
        self._action_meta = None
        self.emit({'kind': 'timing', **self.timing_snapshot()}, record=False)

    def _finish_timing(self) -> None:
        if self._game_started is not None and self._game_finished is None:
            self._game_finished = time.monotonic()
            self.game_finished_at = time.strftime('%Y-%m-%d %H:%M:%S')
        self._action_started = None
        self._action_meta = None
        if self._game_started is not None:
            self.emit({'kind': 'timing', **self.timing_snapshot()}, record=False)

    def timing_payload(self) -> dict[str, Any] | None:
        '''
        Per-game and per-player action timings for the export JSON.

        None when no real game was timed (idle, or a replay), so a consumer can
        tell "no data" from "a game with no decisions".
        '''
        if self._game_started is None:
            return None
        end = self._game_finished if self._game_finished is not None else time.monotonic()
        # Every seat gets a row, including one who never acted (0 actions), so a
        # downstream analysis can treat the table as fixed-size.
        players: dict[int, dict[str, Any]] = {}
        state = self.game.game_state if self.game is not None else None
        if state is not None and state.players:
            for player_id, player in state.players.items():
                players[player_id] = {
                    'id': player_id, 'role': player.job.value,
                    'actions': 0, 'total_seconds': 0.0, 'max_seconds': 0.0,
                }
        for record in self.action_times:
            entry = players.setdefault(record['player'], {
                'id': record['player'], 'role': record.get('role'),
                'actions': 0, 'total_seconds': 0.0, 'max_seconds': 0.0,
            })
            entry['actions'] += 1
            entry['total_seconds'] += record['seconds']
            entry['max_seconds'] = max(entry['max_seconds'], record['seconds'])
        for entry in players.values():
            entry['total_seconds'] = round(entry['total_seconds'], 3)
            entry['max_seconds'] = round(entry['max_seconds'], 3)
            entry['avg_seconds'] = round(entry['total_seconds'] / entry['actions'], 3) \
                if entry['actions'] else 0.0
        return {
            'recorded': True,
            'started_at': self.game_started_at,
            'finished_at': self.game_finished_at,
            'total_seconds': round(max(0.0, end - self._game_started), 3),
            'players': [players[key] for key in sorted(players)],
            'actions': [dict(record) for record in self.action_times],
        }

    def request_answer(self, prompt: str, player: int) -> str:
        meta = dict(self.structured) if self.structured else {'key': None, 'required': False, 'choices': None}
        if self._replay_script is not None:
            # Replaying must never call a model or wait for a browser: it feeds
            # the recorded answers back in, which is what makes it free and exact.
            return self._recorded_answer(prompt, player, meta)
        self._harvest()
        timed = self._begin_action(prompt, player, meta)
        try:
            if self.seats.get(player) == 'llm':
                return self._llm_answer(prompt, player, meta)
            return self._human_answer(prompt, player, meta)
        finally:
            if timed:
                self._end_action()

    def _secret(self, prompt: str, meta: dict[str, Any]) -> bool:
        '''
        True when the question itself gives the role away.

        A statement, a vote, and the political decisions the rules make public
        are all fair game for everyone: the sheriff's election, the badge, the
        speaking order and a dying shot are announced to the whole table anyway.
        "Input the player's ID to poison" identifies the witch, and "do you want
        to expose your identification" would identify a wolf before he has
        decided - those prompts and their answers stay in the private panel.
        '''
        kind = llm_interface.classify(str(meta.get('key') or ''), prompt)
        return kind not in PUBLIC_KINDS

    def _prompt_event(self, prompt: str, player: int, meta: dict[str, Any],
                      controller: str) -> dict[str, Any]:
        # `prompt` is what the operator sees, `raw` is engine.py's own English text.
        # Keeping both lets the transcript be re-rendered on a language switch.
        # `private` goes to the private panel, never the shared transcript: the
        # question names the role ("input the ID to poison"), so publishing it
        # would tell everyone who is acting.
        kind = llm_interface.classify(str(meta.get('key') or ''), prompt)
        return {'kind': 'prompt', **meta, 'player': player, 'controller': controller,
                'prompt': i18n.engine_text(prompt, self.language), 'raw': prompt,
                'secret': self._secret(prompt, meta),
                # the console renders 是/否 buttons from `yesno` instead of a
                # text box, and labels the decision from `decision`
                'decision': kind.value,
                'yesno': kind in llm_interface.YES_NO_KINDS,
                'private': self._private_view(player)}

    def _private_view(self, player: int) -> dict[str, Any]:
        '''
        What only the seat now acting may see.

        A human seat needs its own role and secrets to play at all, and the
        private panel is wiped when that seat answers, so handing the screen to
        the next player does not hand over the previous one's information.

        `camp_mates` is the one thing the board needs as data rather than prose:
        a wolf recognises the rest of the pack, but only as "the wolf camp" -
        the Wolf King is deliberately indistinguishable from a plain Wolf.
        '''
        try:
            observation = self.observation(player)
        except Exception:
            return {'role': None, 'facts': [], 'camp_mates': []}

        camp_mates: list[int] = []
        state = self.game.game_state if self.game is not None else None
        wolf_camp = {job.value for job in engine.GameRule.WolfCamp}
        if state is not None and observation.role in wolf_camp:
            camp_mates = [i for i in engine.GameRuleDefinitions.players_with(
                state, engine.GameRule.WolfCamp) if i != player]

        return {'role': i18n.name(i18n.JOB_NAMES, observation.role, self.language),
                'facts': list(observation.private_facts),
                'camp_mates': camp_mates}

    def _recorded_answer(self, prompt: str, player: int, meta: dict[str, Any]) -> str:
        self.emit(self._prompt_event(prompt, player, meta, 'replay'))
        value = self._take_recorded()
        self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value,
                   'secret': self._secret(prompt, meta)})
        return value

    def _take_recorded(self) -> str:
        if not self._replay_script:
            raise ReplayExhausted
        if self._replay_delay:
            time.sleep(self._replay_delay)
        return self._replay_script.pop(0)

    def _human_answer(self, prompt: str, player: int, meta: dict[str, Any]) -> str:
        self.pending_input = True
        self.emit(self._prompt_event(prompt, player, meta, 'human'))
        try:
            value = self._inbox.get()
        finally:
            self.pending_input = False
        self.answers.append(value)
        self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value,
                   'secret': self._secret(prompt, meta)})
        return value

    def model_for(self, player: int) -> llm_interface.ProviderConfig:
        '''The model a seat plays: its profile if models.json assigns one.'''
        if self.models is None:
            return self.provider_config
        return self.models.config_for(player, self.provider_config)

    def _llm_answer(self, prompt: str, player: int, meta: dict[str, Any]) -> str:
        request = llm_interface.DecisionRequest(
            kind=llm_interface.classify(str(meta.get('key') or ''), prompt),
            question=prompt.strip(),
            allowed=None if meta.get('choices') is None else tuple(meta['choices']),
            required=bool(meta.get('required')),
            key=str(meta.get('key') or ''),
        )
        self.emit(self._prompt_event(prompt, player, meta, 'llm'))
        self.emit({'kind': 'thinking', 'player': player, 'label': request.kind.value})
        try:
            decision = self._controller(player).decide(self.observation(player), request)
        except llm_interface.LLMError as error:
            self.emit({'kind': 'llm-error', 'text': f'P{player}: {error}'})
            decision = llm_interface.LLMController.fallback(
                request, str(error), self.language)
        value = decision.as_answer()
        self.emit({
            'kind': 'decision', 'player': player, 'label': request.kind.value,
            'value': value, 'speech': decision.speech,
            'reasoning': decision.reasoning, 'note': decision.note,
        })
        # Recorded like a human answer, so replaying re-simulates the same game
        # from the same seed without paying for another round of API calls.
        self.answers.append(value)
        self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value,
                   'secret': self._secret(prompt, meta)})
        return value

    def _controller(self, player: int) -> llm_interface.LLMController:
        cached = self.controllers.get(player)
        if cached is None:
            provider = llm_interface.make_provider(self.model_for(player))
            cached = llm_interface.LLMController(
                provider, on_event=self._llm_event, language=self.language)
            self.controllers[player] = cached
        return cached

    def _llm_event(self, event: dict[str, Any]) -> None:
        if not self.show_prompts and event.get('kind') in ('llm-request', 'llm-response'):
            return
        self.emit(event)

    def submit(self, value: str) -> bool:
        # A replay must never consume browser input: the answer would sit in the
        # queue and be handed to the next live game as if it were its own.
        if not self.pending_input or self._replay_script is not None:
            return False
        self._inbox.put(value)
        return True

    def _drain_inbox(self) -> None:
        while not self._inbox.empty():
            self._inbox.get_nowait()

    # -- LLM configuration and seat routing --------------------------------
    def set_seats(self, seats: dict[Any, Any]) -> None:
        for raw_id, controller in seats.items():
            try:
                player_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            # the whole table, not the old six seats: a hardcoded 5 here meant
            # seats 6-11 silently ignored every seat toggle
            if 0 <= player_id < engine.PLAYERS_COUNT:
                self.seats[player_id] = 'llm' if str(controller) == 'llm' else 'human'
        self.emit({'kind': 'seats', 'seats': dict(self.seats)})

    def model_report(self) -> dict[str, Any]:
        """
        What the console may show about the models.

        Model settings live in models.json now, so the page reads them and never
        writes them. Keys never leave the process: this says whether one exists
        and where it came from, not what it is.
        """
        fallback = self.provider_config
        report: dict[str, Any] = {
            'fallback': {'kind': fallback.kind, 'model': fallback.model,
                         'has_key': bool(fallback.api_key)},
            'seats': {str(seat): {'profile': '', 'kind': fallback.kind,
                                  'model': fallback.model}
                      for seat in range(engine.PLAYERS_COUNT)},
            'models': {},
            'default': '',
            'source': '',
            'error': self.model_config_error,
        }
        if self.models is None:
            return report
        described = self.models.describe(fallback)
        report.update({
            'source': described['source'],
            'default': described['default'],
            'models': described['models'],
            'seats': described['seats'],
        })
        return report

    def reload_models(self) -> bool:
        """Re-read models.json, so an edit does not need a server restart."""
        self.controllers.clear()
        if not MODELS_PATH.is_file():
            self.models = None
            self.model_config_error = ''
            return True
        try:
            self.models = llm_interface.load_model_config(MODELS_PATH)
        except llm_interface.LLMError as error:
            # keep the previous registry: a typo must not take the game down
            self.model_config_error = str(error)
            self.emit({'kind': 'llm-error',
                       'text': i18n.t('bad_settings', self.language, error=error)})
            return False
        self.model_config_error = ''
        return True

    def test_llm(self) -> None:
        # On a worker thread: a blocking HTTP call here would stall the event
        # loop that is also delivering events to the browser.
        self.emit({'kind': 'notice', 'text': i18n.t('contacting', self.language)})
        threading.Thread(target=self._test_llm_worker, name='llm-test', daemon=True).start()

    def _distinct_models(self) -> list[tuple[str, llm_interface.ProviderConfig]]:
        """Every model the table would actually call, once each."""
        seen: dict[tuple[str, str, str], tuple[str, llm_interface.ProviderConfig]] = {}
        for seat in range(engine.PLAYERS_COUNT):
            config = self.model_for(seat)
            key = (config.kind, config.base_url, config.model)
            if key not in seen:
                name = self.models.profile_for(seat) if self.models else ''
                seen[key] = (name or config.model, config)
        return list(seen.values())

    def _test_llm_worker(self) -> None:
        '''
        Probe each model with the same prompt shape real decisions use.

        A terse "reply with JSON" probe is not representative: deepseek-chat
        answered one with a placeholder instead of a number, which said nothing
        about whether actual decisions would decode. This asks a real question
        and checks that the reply parses.
        '''
        observation = llm_interface.Observation(
            viewer=0, role='Civilian', stage='Day', cycle=1,
            living=(0, 1, 2, 3, 4, 5), out_of_play=(),
            speeches=((1, i18n.t('probe_speech', self.language)),),
        )
        request = llm_interface.DecisionRequest(
            llm_interface.DecisionKind.Vote, 'Vote whom?', (1, 2, 3, 4, 5), required=False,
        )
        system, user = llm_interface.build_messages(observation, request)
        for name, config in self._distinct_models():
            try:
                provider = llm_interface.make_provider(config)
                raw = provider.complete(system, user)
                decision = llm_interface.decode(raw, request)
            except llm_interface.LLMError as error:
                self.emit({'kind': 'llm-error', 'text': i18n.t(
                    'conn_failed_model', self.language, model=name, error=error)})
                continue
            except llm_interface.DecodeError as error:
                self.emit({'kind': 'llm-error', 'text': i18n.t(
                    'decode_failed', self.language, error=error, raw=raw.strip()[:160])})
                continue
            picked = (i18n.t('choose_pass', self.language) if decision.choice is None
                      else i18n.t('choose_choice', self.language, choice=decision.choice))
            self.emit({'kind': 'notice', 'text': i18n.t(
                'model_ready', self.language, model=name, picked=picked)})
            if self.show_prompts:
                self.emit({'kind': 'llm-response', 'viewer': -1, 'attempt': 1,
                           'text': raw.strip()})

    def set_language(self, code: str) -> bool:
        '''Switch language and re-render the whole transcript.'''
        if not i18n.is_supported(code):
            self.emit({'kind': 'notice',
                       'text': i18n.t('unknown_language', self.language, code=code)})
            return False
        self.language = i18n.set_language(code)
        for controller in self.controllers.values():
            controller.language = self.language
        # Announce first: the client switches its own strings, then rebuilds the
        # transcript from the re-emitted events below.
        self.emit({'kind': 'language', 'value': self.language,
                   'label': i18n.LANGUAGES[self.language]})
        self._retranslate()
        return True

    def _retranslate(self) -> None:
        '''
        Re-render the stored transcript in the current language.

        Engine and prompt text keep their untranslated `raw` form, so the
        backscroll switches too instead of ending up half translated. Model
        output cannot be re-rendered: it is whatever language the model was
        prompted in, so an old reply stays as it was written.
        '''
        rebuilt: list[dict[str, Any]] = []
        for old in self.events:
            if old.get('kind') in ('reset', 'language'):
                continue
            event = dict(old)
            if 'raw' in event:
                if event['kind'] == 'log':
                    event['text'] = i18n.engine_text(event['raw'], self.language)
                elif event['kind'] == 'prompt':
                    event['prompt'] = i18n.engine_text(event['raw'], self.language)
            event['seq'] = len(rebuilt)
            rebuilt.append(event)
        with self._events_lock:
            # A distinct list: emitting the reset below appends to self.events,
            # and iterating the same object would have re-emitted that reset at
            # the end, clearing the transcript the client had just rebuilt.
            self.events = list(rebuilt)
        self.emit({'kind': 'reset'})
        for event in rebuilt:
            self.emit(event, record=False)

    # -- exporting and replaying a game ------------------------------------
    def export_replay(self, name: str = '') -> Path | None:
        '''
        Write this game to a JSON file.

        The file holds both halves of the record: every action (seed plus the
        exact sequence of answers, which is what makes a replay deterministic and
        free of model calls) and the whole transcript with each seat's private
        reasoning.
        '''
        if self.seed is None or not self.answers:
            self.emit({'kind': 'notice',
                       'text': i18n.t('nothing_to_export', self.language)})
            return None

        stamp = time.strftime('%Y%m%d-%H%M%S')
        filename = replay_filename(name, f'{stamp}-seed{self.seed}.json')
        if not filename.endswith('.json'):
            filename += '.json'
        payload = {
            'format': REPLAY_FORMAT,
            'version': REPLAY_VERSION,
            'rule_version': engine.GameRule.Version,
            'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'complete': self.status == 'finished',
            'language': self.language,
            'seed': self.seed,
            'model': {'kind': self.provider_config.kind, 'model': self.provider_config.model},
            'seats': dict(self.seats),
            # Wall-clock and per-decision timings for later analysis. None for a
            # replay, which is not timing real players.
            'timing': self.timing_payload(),
            # the roles and fates belong in the record too: the .json is the
            # replayable half, and the .txt renders this same block as prose
            'result': self.snapshot(),
            'answers': list(self.answers),
            'events': list(self.events),
        }
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        path = REPLAY_DIR / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        extra = ''
        if self.raw_output:
            text_path = path.with_suffix('.txt')
            text_path.write_text(self.render_text(payload), encoding='utf-8')
            extra = i18n.t('exported_text', self.language, file=text_path.name)
        self.emit({'kind': 'notice',
                   'text': i18n.t('exported', self.language, file=path.name,
                                  steps=len(self.answers)) + extra})
        self.emit({'kind': 'replays', 'files': self.list_replays()})
        return path

    def render_text(self, payload: dict[str, Any]) -> str:
        '''
        A human-review copy of an exported game.

        The .json stays the only replayable artefact (it carries the seed and the
        exact answers); this file is prose, because a reviewer wants to read a
        game rather than diff JSON: the seating and result, every seat's role and
        fate, then the whole stream with the private half marked and the models'
        reasoning inline.
        '''
        lang = self.language

        def L(key: str, **kwargs: Any) -> str:
            return i18n.t(key, lang, **kwargs)

        result = payload.get('result') or {}
        seats = payload.get('seats') or {}
        players = result.get('players') or []
        model = payload.get('model') or {}
        winner = result.get('winner')

        lines = [L('text_title'), '']
        lines.append(L('text_exported_at', when=payload.get('exported_at', '')))
        lines.append(L('text_seed', seed=payload.get('seed')))
        lines.append(L('text_rule_version',
                       version=payload.get('rule_version') or '?'))
        lines.append(L('text_model', model=f"{model.get('kind', '')}/{model.get('model', '')}"))
        lines.append(L('text_language', code=lang))
        lines.append(L('text_rounds', stage=result.get('stage', ''),
                       cycles=result.get('cycles', 0),
                       state=L('text_finished') if payload.get('complete') else L('text_unfinished')))
        lines.append(L('text_result', result=i18n.name(i18n.CAMP_NAMES, winner, lang)
                       if winner else L('text_no_result')))
        sheriff = result.get('sheriff')
        lines.append(L('text_sheriff', sheriff=f'P{sheriff}' if sheriff is not None else L('text_absent')))
        lines.append(L('text_badge_state',
                       state=L('text_badge_lost') if result.get('badge_broken') else L('text_badge_whole')))
        lines.append(L('text_exploded', count=result.get('exploded', 0)))
        lines.append(L('text_seats', seats='  '.join(
            f"P{seat}={seats.get(str(seat), seats.get(seat, '?'))}"
            for seat in sorted(seats, key=lambda k: int(k)))))
        lines.append('')

        lines.append(L('text_roles_header'))
        for player in players:
            lines.append(L('text_role_row', seat=player.get('id'),
                           job=i18n.name(i18n.JOB_NAMES, player.get('job'), lang),
                           state=L('text_alive') if player.get('state') == 'Alive'
                           else L('text_out')))
        lines.append('')

        lines.append(L('text_stream_header'))
        lines.append(L('text_legend'))
        # Number the rendered lines only: the event stream also carries bridge
        # bookkeeping (snapshots, seat changes) that a reviewer never needs, and
        # gaps in the numbering would read like missing content.
        index = 0
        for event in payload.get('events') or []:
            kind = event.get('kind')
            if kind not in RENDERED_EVENTS:
                continue
            index += 1
            tag = f'#{index:04d}'
            if kind in ('log', 'private'):
                mark = L('text_public') if kind == 'log' else L('text_private')
                body = (event.get('text') or event.get('raw') or '').strip()
                if body:
                    lines.append(f'{tag} [{mark}] {body}')
            elif kind == 'prompt':
                mark = L('text_private') if event.get('secret') else L('text_public')
                who = L('text_seat_acting', seat=event.get('player'))
                controller = L(f"controller_{event.get('controller', 'human')}")
                question = (event.get('prompt') or event.get('raw') or '').strip()
                lines.append(f"{tag} [{mark}] {who} ({controller}) {L('text_asks')} {question}")
                if event.get('choices') is not None:
                    lines.append(f'{tag}       {L("text_legal")} '
                                 + ' / '.join(str(c) for c in event['choices']))
            elif kind == 'answer':
                mark = L('text_private') if event.get('secret') else L('text_public')
                value = str(event.get('value') or '').strip() or L('text_blank')
                lines.append(f"{tag} [{mark}] {L('text_seat_acting', seat=event.get('player'))} "
                             f"{L('text_answers')} {value}")
            elif kind == 'decision':
                value = str(event.get('value') or '').strip() or L('text_blank')
                head = (f"{tag}       {L('text_decision')} {event.get('label')} -> "
                        f"{value}")
                lines.append(head)
                # "speech" is only a statement when the decision was a speech:
                # a vote or a night action has no public sentence, so a model
                # that fills the field is explaining itself, not talking
                fields = [('speech', 'text_speech' if event.get('label') == 'speech'
                           else 'text_thoughts'),
                          ('reasoning', 'text_reasoning'), ('note', 'text_note')]
                for field, key in fields:
                    value = (event.get(field) or '').strip()
                    if value:
                        lines.append(f'{tag}       {L(key)} {value}')
            elif kind == 'thinking':
                label = L('text_thinking', seat=event.get('player'),
                          kind=event.get('label'))
                lines.append(f'{tag}       {label}')
            elif kind == 'llm-error':
                lines.append(f"{tag}       {L('text_error')} {event.get('text')}")
            elif kind == 'history':
                lines.append(f"{tag}       {L('text_history_dump')}")
                lines.extend('       ' + row for row in (event.get('text') or '').splitlines())
            elif kind == 'notice':
                lines.append(f"{tag} [{L('text_public')}] {(event.get('text') or '').strip()}")
        lines.append('')
        lines.append(L('text_footer', steps=len(payload.get('answers') or [])))
        return '\n'.join(lines) + '\n'

    def list_replays(self) -> list[dict[str, Any]]:
        '''Exported games, newest name first, for the console to offer.'''
        if not REPLAY_DIR.exists():
            return []
        items: list[dict[str, Any]] = []
        for path in sorted(REPLAY_DIR.glob('*.json'), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get('format') != REPLAY_FORMAT:
                continue
            result = payload.get('result') or {}
            items.append({
                'name': path.name,
                'seed': payload.get('seed'),
                'winner': result.get('winner'),
                'cycles': result.get('cycles'),
                'steps': len(payload.get('answers') or []),
                'complete': bool(payload.get('complete')),
                'exported_at': payload.get('exported_at', ''),
                # A file from another rule version stays listed but cannot be
                # replayed: the recorded answers were given to different prompts.
                'rule_version': payload.get('rule_version'),
                'compatible': payload.get('rule_version') == engine.GameRule.Version,
            })
        return items

    def replay_file(self, name: str) -> bool:
        '''
        Re-run a game that was exported earlier.

        Re-simulation rather than playback: the recorded answers are fed back
        through the engine, so the transcript is reproduced by the real rules and
        no model is called.

        Only an export recorded under the current `engine.GameRule.Version` is
        replayed: its answers were given to prompts of that rule set, so a file
        from another version is refused instead of silently producing a
        different game.
        '''
        if self.status in ('running', 'replaying'):
            self.emit({'kind': 'notice', 'text': i18n.t('busy', self.language)})
            return False
        path = REPLAY_DIR / Path(name).name
        if not path.is_file():
            self.emit({'kind': 'notice',
                       'text': i18n.t('replay_missing', self.language, file=name)})
            return False
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            self.emit({'kind': 'notice',
                       'text': i18n.t('replay_bad', self.language, error=error)})
            return False

        answers = [str(value) for value in (payload.get('answers') or [])]
        if payload.get('format') != REPLAY_FORMAT or payload.get('seed') is None or not answers:
            self.emit({'kind': 'notice', 'text': i18n.t('replay_bad', self.language,
                                                        error='missing seed or answers')})
            return False
        recorded = payload.get('rule_version')
        if recorded != engine.GameRule.Version:
            if recorded:
                text = i18n.t('replay_incompatible', self.language, file=path.name,
                              recorded=recorded, current=engine.GameRule.Version)
            else:
                text = i18n.t('replay_no_version', self.language, file=path.name,
                              current=engine.GameRule.Version)
            self.emit({'kind': 'notice', 'text': text})
            return False

        self.seed = int(payload['seed'])
        self.answers = answers
        self.copy = {'file': path.name, 'complete': bool(payload.get('complete')),
                     'exported_at': payload.get('exported_at', '')}
        self.emit({'kind': 'notice', 'text': i18n.t('replaying_file', self.language,
                                                    file=path.name, steps=len(answers))})
        self.start_replay(delay=float(self._replay_delay or 0.0) or REPLAY_DELAY,
                          seed=self.seed)
        return True

    # -- per-seat observation ----------------------------------------------
    def _fact(self, key: str, **params: Any) -> str:
        if 'verdict' in params:
            params['verdict'] = i18n.t(params['verdict'], self.language)
        return i18n.t(key, self.language, **params)

    def _player_signature(self) -> tuple[tuple[int, str], ...]:
        '''
        What the board shows.

        The political state is part of it: electing a sheriff, tearing the badge
        or a self-explosion all change the board without changing any player's
        health, and the board must refresh for those too.
        '''
        state = self.game.game_state if self.game is not None else None
        if state is None or state.players is None:
            return ()
        return tuple([(p.ID, p.player_state.name)
                      for p in sorted(state.players.values(), key=lambda p: p.ID)]
                     + [(-1, str(self.snapshot().get(key)))
                        for key in ('sheriff', 'badge_broken', 'exploded')])

    def _harvest(self) -> None:
        '''
        Fold whatever the engine has newly revealed into per-seat notes.

        Called before every decision. engine.py assigns the Prophet's verdict to an
        attribute on her action object after execute() runs and wipes it at the
        next nightfall, so the only reliable moment to read it is a later prompt.
        '''
        game = self.game
        if game is None or game.game_state.players is None:
            return
        state = game.game_state

        # The console only heard about player states when a game started or
        # ended, so the board kept showing the dead as alive for the whole game.
        # Every decision is preceded by this, which is soon enough to look live.
        signature = self._player_signature()
        if signature != self._seen_players:
            self._seen_players = signature
            self.emit({'kind': 'status', 'state': self.status, 'seed': self.seed,
                       **self.snapshot()})

        exited = set(state.exited_players or ())
        newly = exited - self.exited_seen
        if newly:
            day = state.day
            # At the first prompt of a day the day's messages are still empty, so
            # a jump here is the night's deaths rather than the day's vote.
            phase = 'night' if not (day and day.messages) else 'vote'
            self.exit_log.append((state.cycles, phase, tuple(sorted(newly))))
            self.exited_seen = exited

        lookup = state.job_lookup_table or {}
        for player_id in lookup.get(engine.Jobs.Prophet, ()):
            action = state.players[player_id].job_actions
            target = getattr(action, 'target', None)
            verdict = getattr(action, 'target_attribute', None)
            if target is None or verdict is None:
                continue
            # stored as a template plus parameters so the notes survive a
            # language switch instead of freezing in one language
            fact = ('fact_divination', {'target': target,
                                        'verdict': 'verdict_wolf' if verdict else 'verdict_clean'})
            notes = self.memory.setdefault(player_id, [])
            if fact not in notes:
                notes.append(fact)

    def _render_history(self) -> tuple[str, ...]:
        '''
        Reuse the engine's own round history as context.

        Only the public half is safe to hand to every seat: DayHistory carries
        the day's statements and the vote tally, whereas NightHistory holds the
        Wolves' target list and the Prophet's query, which would leak the whole
        game if it were passed through verbatim.
        '''
        lang = self.language
        nothing = i18n.t('history_none', lang)
        joiner = '，' if lang == 'zh' else ', '
        lines = []
        for index, (_, day) in enumerate(self.game.game_state.history or (), start=1):
            said = ' | '.join(i18n.t('history_said', lang, player=p, text=text)
                              for p, text in sorted((day.messages or {}).items()))
            tally = joiner.join(i18n.t('history_votes', lang, player=p, count=votes[0])
                                for p, votes in sorted((day.vote_result or {}).items()) if votes[0])
            left = [ids for cycle, _, ids in self.exit_log if cycle == index]
            left_text = joiner.join(i18n.player_ids(ids, lang) for ids in left) \
                or i18n.t('history_nobody', lang)
            lines.append(i18n.t('history_round', lang, round=index, said=said or nothing,
                                votes=tally or nothing, left=left_text))
        return tuple(lines)

    def observation(self, viewer: int) -> llm_interface.Observation:
        '''The world as one seat is allowed to see it. Never pass raw state on.'''
        state = self.game.game_state
        players = state.players
        role = players[viewer].job.value
        day, night = state.day, state.night

        # engine.py only resets DayHistory when the Day stage is entered, so at
        # night `state.day` still holds the day that just ended. Reporting that
        # as "today's statements" and "today's tally" made models reason about a
        # vote that had already closed - one wolf read yesterday's tally as live
        # and talked about a player who had already been voted out. The finished
        # day is still visible under the round history.
        is_day = state.stage is engine.GameStage.Day
        speeches = tuple(sorted((day.messages or {}).items())) if is_day and day else ()
        tally = tuple(sorted(
            (k, v[0]) for k, v in (day.vote_result or {}).items() if v[0]
        )) if is_day and day else ()
        nights = [ids for _, phase, ids in self.exit_log if phase == 'night']
        fallen = nights[-1] if nights else ()

        facts = [self._fact(key, **params) for key, params in self.memory.get(viewer, ())]
        # The whole wolf camp shares the pack's knowledge, so the Wolf King and
        # the wolves are told about each other
        if role in {job.value for job in engine.GameRule.WolfCamp}:
            pack = engine.GameRuleDefinitions.players_with(state, engine.GameRule.WolfCamp)
            mates = [i for i in pack if i != viewer]
            if mates:
                facts.append(self._fact('fact_fellow_wolf',
                                        ids=i18n.player_ids(mates, self.language)))
            if night and night.wolf_kills:
                facts.append(self._fact('fact_pack_named',
                                        ids=i18n.player_ids(night.wolf_kills, self.language)))
            # The pack discusses in seat order, so a wolf speaking later sees
            # what its packmates have proposed. Their final victim is not shown:
            # it is only settled after the last wolf has spoken.
            earlier = [(wolf, target) for wolf, target in
                       sorted((night.wolf_proposals or {}).items() if night else ())
                       if wolf != viewer]
            if earlier:
                joiner = '；' if self.language == 'zh' else '; '
                items = joiner.join(
                    self._fact('fact_pack_proposal_pass', wolf=wolf) if target is None
                    else self._fact('fact_pack_proposal_target', wolf=wolf, target=target)
                    for wolf, target in earlier)
                facts.append(self._fact('fact_pack_proposals', items=items))
        elif role == 'Guard':
            # 不能连守 is public knowledge once it has happened, but remembering
            # whom you protected is the guard's own bookkeeping.
            if state.guarded_players:
                facts.append(self._fact('fact_guarded_before',
                                        ids=i18n.player_ids(state.guarded_players, self.language)))
        elif role == 'Witch':
            # The potions are one per game and the counts live on her action
            # object, so without this a model has no way to know the limit.
            action = players[viewer].job_actions
            poison = getattr(action, 'poison_left', None)
            antidote = getattr(action, 'antidote_left', None)
            if poison is not None and antidote is not None:
                facts.append(self._fact('fact_potions', poison=poison, antidote=antidote))
            if night and night.wolf_kills:
                facts.append(self._fact('fact_wolves_killed',
                                        ids=i18n.player_ids(night.wolf_kills, self.language)))
            if night and night.witch_killed_target is not None:
                facts.append(self._fact('fact_you_killed', player=night.witch_killed_target))
            if night and night.witch_saved_target is not None:
                facts.append(self._fact('fact_you_saved', player=night.witch_saved_target))

        # The election exists only on day 1, and `day` carries the sheriff from
        # the previous day once the new one has begun.
        poll = (day.police_vote_person_camp or {}) if day else {}
        candidates = tuple(poll.get(True, ())) if poll else ()
        candidate_speeches = tuple(sorted((day.candidate_speech or {}).items())) if day else ()
        sheriff = day.the_police if day else None
        # A sheriff who has left play cannot act; report no sheriff instead of
        # handing a model a stale seat that it might try to reason about.
        if sheriff is not None and players[sheriff].player_state is engine.PlayerState.Exit:
            sheriff = None

        return llm_interface.Observation(
            viewer=viewer,
            role=role,
            stage=state.stage.name,
            cycle=state.cycles,
            living=tuple(sorted(p.ID for p in players.values()
                                if p.player_state is engine.PlayerState.Alive)),
            out_of_play=tuple(sorted(state.exited_players or ())),
            fallen_last_night=fallen,
            speeches=speeches,
            earlier_rounds=self._render_history(),
            private_facts=tuple(facts),
            vote_tally=tally,
            sheriff=sheriff,
            sheriff_badge_broken=bool(day.break_the_police) if day else False,
            sheriff_candidates=candidates,
            candidate_speeches=candidate_speeches,
            exploded_wolves=len(state.wolf_exposed or ()),
        )

    # -- state -------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        state = self.game.game_state if self.game is not None else None
        if state is None or state.players is None:
            return {'stage': 'Initializing', 'cycles': 1, 'exited': [], 'players': [], 'winner': None}
        day = state.day
        sheriff = day.the_police if day else None
        if sheriff is not None and state.players[sheriff].player_state is engine.PlayerState.Exit:
            sheriff = None
        return {
            'stage': state.stage.name,
            'cycles': state.cycles,
            'exited': list(state.exited_players or []),
            'winner': WINNERS.get(state.stage.name),
            'sheriff': sheriff,
            'badge_broken': bool(day.break_the_police) if day else False,
            'candidates': list((day.police_vote_person_camp or {}).get(True, ())) if day else [],
            'exploded': len(state.wolf_exposed or ()),
            'players': [
                {'id': p.ID, 'job': p.job.value, 'state': p.player_state.name}
                for p in sorted(state.players.values(), key=lambda p: p.ID)
            ],
        }

    def public_state(self) -> dict[str, Any]:
        return {
            'kind': 'snapshot',
            'status': self.status,
            'seed': self.seed,
            'message': self.message,
            'events': list(self.events),
            'seats': dict(self.seats),
            'replays': self.list_replays(),
            'language': self.language,
            'languages': i18n.LANGUAGES,
            'llm': self.provider_config.describe(),
            'models': self.model_report(),
            'llm_error': self.config_error,
            'show_prompts': self.show_prompts,
            'reveal_jobs': self.reveal_jobs,
            'show_thinking': self.show_thinking,
            'raw_output': self.raw_output,
            'timing': self.timing_snapshot(),
            **self.snapshot(),
        }

    # -- driving the engine ------------------------------------------------
    def start_game(self, seed: int | None = None) -> None:
        if self.status in ('running', 'replaying'):
            self.emit({'kind': 'notice', 'text': i18n.t('busy', self.language)})
            return
        self.seed = random.randrange(1 << 30) if seed is None else int(seed)
        self.events.clear()
        self.answers.clear()
        self.game = None
        self.memory.clear()
        self.exit_log.clear()
        self.exited_seen.clear()
        self._seen_players = ()
        self._drain_inbox()
        # A real game is timed from the moment it starts; a replay never is.
        self._reset_timing()
        self.game_started_at = time.strftime('%Y-%m-%d %H:%M:%S')
        self._game_started = time.monotonic()
        self.emit({'kind': 'reset'})
        self._spawn(replay_script=None, delay=0.0, status='running')
        self.emit({'kind': 'timing', **self.timing_snapshot()}, record=False)

    def start_replay(self, delay: float = 0.3, seed: int | None = None) -> None:
        if self.status in ('running', 'replaying'):
            self.emit({'kind': 'notice', 'text': i18n.t('busy', self.language)})
            return
        script = list(self.answers)
        if not script:
            self.emit({'kind': 'notice', 'text': i18n.t('nothing_to_replay', self.language)})
            return
        if seed is not None:
            self.seed = int(seed)
        self.events.clear()
        # Replaying is not a timed game: hide any clock from the last one.
        self._reset_timing()
        self.emit({'kind': 'reset'})
        self._spawn(replay_script=script, delay=max(0.0, float(delay)), status='replaying')
        self.emit({'kind': 'timing', **self.timing_snapshot()}, record=False)

    def _spawn(self, replay_script: list[str] | None, delay: float, status: str) -> None:
        self._replay_script = replay_script
        self._replay_delay = delay
        self.status = status
        self.message = ''
        self.thread = threading.Thread(target=self._run_engine, name='engine', daemon=True)
        self.thread.start()

    def _run_engine(self) -> None:
        assert self.seed is not None
        try:
            # Seeding makes a game a pure function of (seed, answers), which is
            # what lets the replay below reproduce it exactly.
            random.seed(self.seed)
            game = engine.Game()
            game.init_game()
            self.game = game
            self.emit({'kind': 'status', 'state': self.status, 'seed': self.seed, **self.snapshot()})
            game.start()
            self.status = 'finished'
            winner = self.snapshot()['winner']
            if winner and winner != 'Tie':
                message = i18n.t('win_message', self.language,
                                 camp=i18n.name(i18n.CAMP_NAMES, winner, self.language))
            else:
                message = i18n.t('ended_tie', self.language)
            self.emit({
                'kind': 'status', 'state': 'finished', 'seed': self.seed,
                'message': message, **self.snapshot(),
            })
        except ReplayExhausted:
            self.status = 'finished'
            self.emit({'kind': 'status', 'state': 'finished',
                       'message': i18n.t('replay_ended', self.language), **self.snapshot()})
        except Exception as exc:
            self.status = 'error'
            self.message = f'{type(exc).__name__}: {exc}'
            self.emit({'kind': 'status', 'state': 'error', 'message': self.message, **self.snapshot()})
            self.emit({'kind': 'log', 'text': traceback.format_exc()})
        finally:
            self._finish_timing()
            self._replay_script = None
            self._replay_delay = 0.0


# -- compatibility shims ---------------------------------------------------
# engine.py exposes no UI hooks of its own, so these rebind module-level names
# from the outside. Each one names the core line that makes it necessary.
CURRENT: Session | None = None

_original_input_an_id = engine.JobAction.input_an_id
_real_print = builtins.print
_real_pprint = pprint_module.pprint


def _engine_thread() -> bool:
    return CURRENT is not None and threading.current_thread() is CURRENT.thread


def _input_an_id(prompt, required, player_id, key='ID', exclude=()):
    # Metadata is published *around* the original call so the UI can offer the
    # legal IDs, while the original still does the validating and re-prompting.
    if CURRENT is not None:
        CURRENT.begin_structured_input(key, required, exclude)
    try:
        return _original_input_an_id(prompt, required, player_id, key, exclude)
    finally:
        if CURRENT is not None:
            CURRENT.end_structured_input()


def _game_input(prompt, player):
    if CURRENT is None:
        raise RuntimeError('no active session')
    return CURRENT.request_answer(prompt, player)


def _game_print(*args, sep=' ', end='\n'):
    if _engine_thread():
        text = sep.join(str(a) for a in args)
        CURRENT.emit({'kind': 'log', 'raw': text,
                      'text': i18n.engine_text(text, CURRENT.language)})
    else:
        _real_print('[Game]', *args, sep=sep, end=end)


def _game_print_private(*args, sep=' ', end='\n'):
    # engine.py's game_print_private: only the seat now acting may see it, so it
    # goes to the private panel and is wiped when that seat is done.
    if _engine_thread():
        text = sep.join(str(a) for a in args)
        CURRENT.emit({'kind': 'private', 'raw': text,
                      'text': i18n.engine_text(text, CURRENT.language)})
    else:
        _real_print('[Game] [Private]', *args, sep=sep, end=end)


def _print(*args, **kwargs):
    # The day's message summary uses a bare print rather than game_print, and no
    # module-level name can intercept a builtin.
    if _engine_thread():
        text = ' '.join(str(a) for a in args)
        CURRENT.emit({'kind': 'log', 'raw': text,
                      'text': i18n.engine_text(text, CURRENT.language)})
    else:
        _real_print(*args, **kwargs)


def _pprint(obj, stream=None, *args, **kwargs):
    # The end-of-game history dump goes straight to stdout; show it in the transcript
    # instead of the server console.
    if _engine_thread():
        CURRENT.emit({'kind': 'history', 'text': pprint_module.pformat(obj)})
    else:
        _real_pprint(obj, stream, *args, **kwargs)


class _PprintProxy:
    pprint = staticmethod(_pprint)


engine.JobAction.input_an_id = staticmethod(_input_an_id)
engine.game_print = _game_print
engine.game_print_private = _game_print_private
engine.game_input = _game_input
engine.clear_screen = lambda: None      # os.system('clear') would spew escapes
engine.pprint = _PprintProxy()
builtins.print = _print


# -- web app ---------------------------------------------------------------
app = FastAPI(title='Werewolf operator console')


@app.get('/')
async def index() -> HTMLResponse:
    # Read per request so edits to the page show up on reload without a restart.
    return HTMLResponse(PAGE_PATH.read_text(encoding='utf-8'))


def handle(message: dict[str, Any]) -> None:
    assert CURRENT is not None
    match message.get('type'):
        case 'start':
            raw = str(message.get('seed') or '').strip()
            CURRENT.start_game(int(raw) if raw else None)
        case 'answer':
            if not CURRENT.submit(str(message.get('value', ''))):
                CURRENT.emit({'kind': 'notice',
                              'text': i18n.t('stray_answer', CURRENT.language)})
        case 'replay':
            raw = str(message.get('seed') or '').strip()
            CURRENT.start_replay(delay=float(message.get('delay', 0.3)), seed=int(raw) if raw else None)
        case 'seats':
            CURRENT.set_seats(message.get('seats') or {})
        case 'llm-test':
            CURRENT.test_llm()
        case 'models-reload':
            if CURRENT.reload_models():
                CURRENT.emit({'kind': 'models', 'models': CURRENT.model_report()})
        case 'export':
            CURRENT.export_replay(str(message.get('name') or ''))
        case 'replay-file':
            CURRENT.replay_file(str(message.get('name') or ''))
        case 'language':
            CURRENT.set_language(str(message.get('value') or ''))
        case 'options':
            CURRENT.show_prompts = bool(message.get('show_prompts', CURRENT.show_prompts))
            CURRENT.reveal_jobs = bool(message.get('reveal_jobs', CURRENT.reveal_jobs))
            CURRENT.raw_output = bool(message.get('raw_output', CURRENT.raw_output))
            CURRENT.show_thinking = bool(message.get('show_thinking', CURRENT.show_thinking))
            CURRENT.emit({'kind': 'options', 'show_prompts': CURRENT.show_prompts,
                          'reveal_jobs': CURRENT.reveal_jobs,
                          'show_thinking': CURRENT.show_thinking,
                          'raw_output': CURRENT.raw_output})
        case other:
            CURRENT.emit({'kind': 'notice',
                          'text': i18n.t('unknown_message', CURRENT.language, what=repr(other))})


@app.websocket('/ws')
async def operator(ws: WebSocket) -> None:
    assert CURRENT is not None
    await ws.accept()
    CURRENT.loop = asyncio.get_running_loop()
    CURRENT.clients.add(ws)
    await ws.send_json(CURRENT.public_state())
    try:
        while True:
            handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        CURRENT.clients.discard(ws)


if __name__ == '__main__':
    CURRENT = Session()
    print(f'Werewolf operator console on http://{HOST}:{PORT}')
    uvicorn.run(app, host=HOST, port=PORT, log_level='info')
