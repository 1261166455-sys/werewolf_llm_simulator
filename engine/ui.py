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
REPLAY_FORMAT = 'werewolf-replay'
REPLAY_VERSION = 1
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
        self.seats: dict[int, str] = {i: 'human' for i in range(6)}
        try:
            self.provider_config = llm_interface.ProviderConfig.from_env()
            self.config_error = ''
        except llm_interface.LLMError as error:
            self.provider_config = llm_interface.ProviderConfig()
            self.config_error = str(error)
        self.controller: llm_interface.LLMController | None = None
        self.show_prompts = False
        # Presentational only: the models' own observations are always filtered
        # per seat, so revealing roles here never changes what a model knows.
        self.reveal_jobs = False
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
        # range(6) mirrors the hardcoded 0..5 range that JobAction.input_an_id
        # asserts, so the choices offered here are exactly what it will accept
        self.structured = {
            'key': key,
            'required': bool(required),
            'choices': [i for i in range(6) if i not in exclude],
        }

    def end_structured_input(self) -> None:
        self.structured = None

    def request_answer(self, prompt: str, player: int) -> str:
        meta = dict(self.structured) if self.structured else {'key': None, 'required': False, 'choices': None}
        if self._replay_script is not None:
            # Replaying must never call a model or wait for a browser: it feeds
            # the recorded answers back in, which is what makes it free and exact.
            return self._recorded_answer(prompt, player, meta)
        self._harvest()
        if self.seats.get(player) == 'llm':
            return self._llm_answer(prompt, player, meta)
        return self._human_answer(prompt, player, meta)

    def _prompt_event(self, prompt: str, player: int, meta: dict[str, Any],
                      controller: str) -> dict[str, Any]:
        # `prompt` is what the operator sees, `raw` is engine.py's own English text.
        # Keeping both lets the transcript be re-rendered on a language switch.
        return {'kind': 'prompt', **meta, 'player': player, 'controller': controller,
                'prompt': i18n.engine_text(prompt, self.language), 'raw': prompt}

    def _recorded_answer(self, prompt: str, player: int, meta: dict[str, Any]) -> str:
        self.emit(self._prompt_event(prompt, player, meta, 'replay'))
        value = self._take_recorded()
        self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value})
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
        self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value})
        return value

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
            decision = self._controller().decide(self.observation(player), request)
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
        self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value})
        return value

    def _controller(self) -> llm_interface.LLMController:
        if self.controller is None:
            provider = llm_interface.make_provider(self.provider_config)
            self.controller = llm_interface.LLMController(
                provider, on_event=self._llm_event, language=self.language)
        return self.controller

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
            if 0 <= player_id <= 5:
                self.seats[player_id] = 'llm' if str(controller) == 'llm' else 'human'
        self.emit({'kind': 'seats', 'seats': dict(self.seats)})

    def configure_llm(self, config: dict[str, Any]) -> bool:
        current = self.provider_config
        try:
            self.provider_config = llm_interface.ProviderConfig(
                kind=str(config.get('kind') or current.kind),
                base_url=str(config.get('base_url', current.base_url)),
                model=str(config.get('model', current.model)),
                api_key=str(config.get('api_key') or current.api_key),
                temperature=float(config.get('temperature', current.temperature)),
                max_tokens=int(config.get('max_tokens', current.max_tokens)),
                json_mode=bool(config.get('json_mode', current.json_mode)),
            )
        except (llm_interface.LLMError, TypeError, ValueError) as error:
            self.emit({'kind': 'llm-error',
                       'text': i18n.t('bad_settings', self.language, error=error)})
            return False
        self.controller = None          # settings changed, so rebuild the client
        self.config_error = ''
        # describe() carries its own 'kind' key, so nest it rather than spreading
        # it into the envelope - it would otherwise overwrite the event kind.
        self.emit({'kind': 'llm-config', 'config': self.provider_config.describe()})
        return True

    def test_llm(self) -> None:
        # On a worker thread: a blocking HTTP call here would stall the event
        # loop that is also delivering events to the browser.
        self.emit({'kind': 'notice', 'text': i18n.t('contacting', self.language)})
        threading.Thread(target=self._test_llm_worker, name='llm-test', daemon=True).start()

    def _test_llm_worker(self) -> None:
        '''
        Probe with the same prompt shape real decisions use.

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
        try:
            provider = llm_interface.make_provider(self.provider_config)
            raw = provider.complete(*llm_interface.build_messages(observation, request))
        except llm_interface.LLMError as error:
            self.emit({'kind': 'llm-error',
                       'text': i18n.t('conn_failed', self.language, error=error)})
            return
        try:
            decision = llm_interface.decode(raw, request)
        except llm_interface.DecodeError as error:
            self.emit({'kind': 'llm-error', 'text': i18n.t(
                'decode_failed', self.language, error=error, raw=raw.strip()[:160])})
            return
        picked = (i18n.t('choose_pass', self.language) if decision.choice is None
                  else i18n.t('choose_choice', self.language, choice=decision.choice))
        self.emit({'kind': 'notice',
                   'text': i18n.t('model_ready', self.language, picked=picked)})
        if self.show_prompts:
            self.emit({'kind': 'llm-response', 'viewer': -1, 'attempt': 1, 'text': raw.strip()})

    def set_language(self, code: str) -> bool:
        '''Switch language and re-render the whole transcript.'''
        if not i18n.is_supported(code):
            self.emit({'kind': 'notice',
                       'text': i18n.t('unknown_language', self.language, code=code)})
            return False
        self.language = i18n.set_language(code)
        if self.controller is not None:
            self.controller.language = self.language
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
            'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'complete': self.status == 'finished',
            'language': self.language,
            'seed': self.seed,
            'model': {'kind': self.provider_config.kind, 'model': self.provider_config.model},
            'seats': dict(self.seats),
            'result': {k: v for k, v in self.snapshot().items() if k != 'players'},
            'answers': list(self.answers),
            'events': list(self.events),
        }
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        path = REPLAY_DIR / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
        self.emit({'kind': 'notice',
                   'text': i18n.t('exported', self.language, file=path.name,
                                  steps=len(self.answers))})
        self.emit({'kind': 'replays', 'files': self.list_replays()})
        return path

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
            })
        return items

    def replay_file(self, name: str) -> bool:
        '''
        Re-run a game that was exported earlier.

        Re-simulation rather than playback: the recorded answers are fed back
        through the engine, so the transcript is reproduced by the real rules and
        no model is called.
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
        state = self.game.game_state if self.game is not None else None
        if state is None or state.players is None:
            return ()
        return tuple((p.ID, p.player_state.name)
                     for p in sorted(state.players.values(), key=lambda p: p.ID))

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
        lookup = state.job_lookup_table or {}
        if role == 'Wolf':
            mates = [i for i in lookup.get(engine.Jobs.Wolf, ()) if i != viewer]
            if mates:
                facts.append(self._fact('fact_fellow_wolf',
                                        ids=i18n.player_ids(mates, self.language)))
            if night and night.wolf_kills:
                facts.append(self._fact('fact_pack_named',
                                        ids=i18n.player_ids(night.wolf_kills, self.language)))
        elif role == 'Witch':
            if night and night.wolf_kills:
                facts.append(self._fact('fact_wolves_killed',
                                        ids=i18n.player_ids(night.wolf_kills, self.language)))
            if night and night.witch_killed_target is not None:
                facts.append(self._fact('fact_you_killed', player=night.witch_killed_target))
            if night and night.witch_saved_target is not None:
                facts.append(self._fact('fact_you_saved', player=night.witch_saved_target))

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
        )

    # -- state -------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        state = self.game.game_state if self.game is not None else None
        if state is None or state.players is None:
            return {'stage': 'Initializing', 'cycles': 1, 'exited': [], 'players': [], 'winner': None}
        return {
            'stage': state.stage.name,
            'cycles': state.cycles,
            'exited': list(state.exited_players or []),
            'winner': WINNERS.get(state.stage.name),
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
            'llm_error': self.config_error,
            'show_prompts': self.show_prompts,
            'reveal_jobs': self.reveal_jobs,
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
        self.emit({'kind': 'reset'})
        self._spawn(replay_script=None, delay=0.0, status='running')

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
        self.emit({'kind': 'reset'})
        self._spawn(replay_script=script, delay=max(0.0, float(delay)), status='replaying')

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
            self._replay_script = None
            self._replay_delay = 0.0


# -- compatibility shims ---------------------------------------------------
# engine.py exposes no UI hooks of its own, so these rebind module-level names
# from the outside. Each one names the core line that makes it necessary.
CURRENT: Session | None = None

_original_init_game = engine.Game.init_game
_original_input_an_id = engine.JobAction.input_an_id
_real_print = builtins.print
_real_pprint = pprint_module.pprint


def _engine_thread() -> bool:
    return CURRENT is not None and threading.current_thread() is CURRENT.thread


def _init_game(self, *args, **kwargs):
    _original_init_game(self, *args, **kwargs)
    # engine.py appends to GameState.history at the end of every day, but the field
    # defaults to None and is never assigned, so the first full cycle would raise
    # AttributeError without this.
    self.game_state.history = []
    # _at_night reads a module-level `game` that only engine.py's `__main__` block
    # defines, so importing engine.py as a module leaves it undefined.
    engine.game = self


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


engine.Game.init_game = _init_game
engine.JobAction.input_an_id = staticmethod(_input_an_id)
engine.game_print = _game_print
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
        case 'llm':
            CURRENT.configure_llm(message.get('config') or {})
        case 'llm-test':
            CURRENT.test_llm()
        case 'export':
            CURRENT.export_replay(str(message.get('name') or ''))
        case 'replay-file':
            CURRENT.replay_file(str(message.get('name') or ''))
        case 'language':
            CURRENT.set_language(str(message.get('value') or ''))
        case 'options':
            CURRENT.show_prompts = bool(message.get('show_prompts', CURRENT.show_prompts))
            CURRENT.reveal_jobs = bool(message.get('reveal_jobs', CURRENT.reveal_jobs))
            CURRENT.emit({'kind': 'options', 'show_prompts': CURRENT.show_prompts,
                          'reveal_jobs': CURRENT.reveal_jobs})
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
