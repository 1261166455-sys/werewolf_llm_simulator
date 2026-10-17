'''
Shared helpers for the engine test suite.

Two styles of test live here:

  * fixture tests build a GameState by hand and call one rule function, which
    makes a specific rule easy to pin down
  * game tests run whole games through Driver, which answers every prompt with
    a legal choice and records what was asked

Run everything with:  ./venv/bin/python -m unittest discover -s tests -v
'''
from __future__ import annotations

import random
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

import engine


class Driver:
    '''
    Stands in for the players.

    Every prompt is answered with a choice from the legal set the engine passes
    as `exclude`, which is exactly what the console offers a model. Prompts,
    exits and speeches are recorded so a test can assert on them afterwards.
    '''

    def __init__(self, pass_rate: float = 0.2, yes_rate: float = 0.5):
        self.pass_rate = pass_rate
        self.yes_rate = yes_rate
        # both are replaceable, so a test can force one particular decision
        # (key, legal, voter, prompt) -> a choice from `legal`
        self.pick = lambda key, legal, voter, prompt: random.choice(legal)
        # Self-explosion has its own tests; defaulting to "no" keeps an ordinary
        # game running long enough to cover the later days.
        self.yesno = lambda prompt, player: (
            'n' if 'expose' in prompt
            else ('y' if random.random() < self.yes_rate else 'n'))
        self.prompts: list[tuple[str, int, int, str]] = []   # text, player, cycle, state
        self.legal: dict[str, list[int]] = {}
        self.exits: list[tuple[int, str]] = []
        self.spoke: list[int] = []
        self.state: engine.GameState | None = None
        self.pending = {'v': ''}
        self.queue: list[str] = []      # scripted answers, for testing rejection paths
        self._orig: dict = {}

    # -- installation ------------------------------------------------------
    def __enter__(self) -> "Driver":
        self.install()
        return self

    def __exit__(self, *exc_info) -> None:
        self.uninstall()

    def install(self) -> None:
        self._orig = {
            'print': engine.game_print,
            'print_private': engine.game_print_private,
            'input': engine.game_input,
            'an_id': engine.JobAction.input_an_id,
            'exit': engine.GameRuleDefinitions.exit_player,
            'clear': engine.clear_screen,
        }
        engine.game_print = lambda *a, **k: None
        engine.game_print_private = lambda *a, **k: None
        engine.clear_screen = lambda: None
        engine.game_input = self._game_input
        engine.JobAction.input_an_id = staticmethod(self._input_an_id)
        engine.GameRuleDefinitions.exit_player = staticmethod(self._exit_player)

    def uninstall(self) -> None:
        engine.game_print = self._orig['print']
        engine.game_print_private = self._orig['print_private']
        engine.game_input = self._orig['input']
        engine.JobAction.input_an_id = self._orig['an_id']
        engine.GameRuleDefinitions.exit_player = self._orig['exit']
        engine.clear_screen = self._orig['clear']

    # -- the two input paths ----------------------------------------------
    def _game_input(self, prompt: str, player: int) -> str:
        state = self.state
        cycle = state.cycles if state is not None else 0
        status = state.players[player].player_state.name if state and state.players else '?'
        self.prompts.append((prompt.strip(), player, cycle, status))
        if 'y/n' in prompt:
            answer = self.yesno(prompt, player)
            # the engine reads a string; accepting a bool keeps tests readable
            return ('y' if answer else 'n') if isinstance(answer, bool) else answer
        if prompt.strip() in ('[Addresser]', '[Speecher]'):
            self.spoke.append(player)
            return f'statement by P{player}'
        if self.queue:
            return self.queue.pop(0)
        return self.pending['v']

    def _input_an_id(self, prompt, required, player_id, key='ID', exclude=()):
        legal = [i for i in range(engine.PLAYERS_COUNT) if i not in exclude]
        self.legal[key] = legal
        if not legal:
            self.pending['v'] = ''
            return {}
        if not required and random.random() < self.pass_rate:
            self.pending['v'] = ''
        else:
            self.pending['v'] = str(self.pick(key, legal, player_id, prompt))
        return self._orig['an_id'](prompt, required, player_id, key, exclude)

    def _exit_player(self, state, player_id, reason=engine.OutReason.Not_Out_Yet):
        if state.players[player_id].player_state is engine.PlayerState.Exit:
            return
        self.exits.append((player_id, reason.name))
        return self._orig['exit'](state, player_id, reason)

    # -- running ----------------------------------------------------------
    def reset_recordings(self) -> None:
        self.prompts.clear()
        self.exits.clear()
        self.spoke.clear()
        self.legal.clear()

    def play(self, seed: int = 0, pass_rate: float | None = None) -> engine.GameState:
        if pass_rate is not None:
            self.pass_rate = pass_rate
        random.seed(seed)
        self.reset_recordings()
        game = engine.Game()
        game.init_game()
        self.state = game.game_state
        game.start()
        return game.game_state

    def shoot_prompts(self) -> int:
        return sum(1 for text, _, _, _ in self.prompts if 'shoot' in text)


# -- fixture helpers -------------------------------------------------------
def new_state(cycle: int = 2, stage: engine.GameStage = engine.GameStage.Night,
              jobs: dict[int, engine.Jobs] | None = None) -> engine.GameState:
    '''
    A hand-built state: everyone a civilian unless `jobs` says otherwise.

    job_lookup_table is maintained so rule functions can query it, which is the
    fixture mistake that makes a test pass for the wrong reason.
    '''
    state = engine.GameState(stage)
    state.cycles = cycle
    state.players = {i: engine.Player(i) for i in range(engine.PLAYERS_COUNT)}
    state.exited_players = []
    state.guarded_players = []
    state.wolf_exposed = []
    state.history = []
    state.job_lookup_table = {job: [] for job in engine.Jobs
                              if job is not engine.Jobs.Not_Assigned}
    for player in state.players.values():
        player.job = engine.Jobs.Civilian
        player.job_actions = engine.CivilianAction()
        state.job_lookup_table[engine.Jobs.Civilian].append(player.ID)

    for player_id, job in (jobs or {}).items():
        state.players[player_id].job = job
        state.players[player_id].job_actions = engine.JOB_ACTIONS[job]()
        state.job_lookup_table[engine.Jobs.Civilian].remove(player_id)
        state.job_lookup_table[job].append(player_id)

    state.night = engine.NightHistory(wolf_kills=[])
    state.day = engine.DayHistory(messages={}, police_vote_person_camp={True: [], False: []},
                                  candidate_speech={}, deaths_comment=[],
                                  wolf_exposed_this_day=[],
                                  the_police=None, break_the_police=None)
    return state


def kill(state: engine.GameState, player_id: int) -> None:
    '''Mark a player Dead, as the night rounds do before the dawn.'''
    state.players[player_id].player_state = engine.PlayerState.Dead


def exited(state: engine.GameState, *player_ids: int) -> None:
    for player_id in player_ids:
        state.players[player_id].player_state = engine.PlayerState.Exit
        state.exited_players.append(player_id)
