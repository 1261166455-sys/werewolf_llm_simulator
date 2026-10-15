import os
import random
import pprint

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import colorama

game_print = lambda *args, sep=' ', end='\n': print('[Game]', *args, sep=sep, end=end) 
game_input = lambda prompt, player: input(f'[Game] [Player {player} operate] {prompt}')


MAXIMUM_CYCLES = 10

def clear_screen():
    if os.system('clear'):
        os.system('cls')

class GameException(Exception):
    pass


class Jobs(Enum):
    '''
    Types of job in our simple game
    '''
    # Also identified by index 0-3
    Civilian = 'Civilian' # 2
    Wolf = 'Wolf'         # 2
    Witch = 'Witch'       # 1
    Prophet = 'Prophet'   # 1
    Not_Assigned = 'Unknown' 

type Response = dict[str, Any]

class JobAction(ABC):

    @staticmethod
    def input_an_id(prompt: str, required: bool, player_id: int, key: str = 'ID', exclude: tuple = ()) -> Response:
        ID = ''
        while True: 
            ID_ = game_input(prompt, player_id)
            if not required and not ID_:
                    return {}
            
            try:
                ID = int(ID_)
                assert 0 <= ID <= 5 and ID not in exclude, 'Invalid player ID'
                break
            except ValueError:
                game_print('Empty or invalid format, please re-input')
                continue
            except AssertionError:
                game_print('Invalid target chosen, please re-input')
                continue

        return { key: ID }

    def provide(self, *args, **kw) -> None:
        pass

    def comment(self, game: GameState, player: Player) -> None:
        game.day.messages[player.ID] = game_input(f'[Addresser] ', player.ID)

    '''
    Actions for a job, a interface
    
    Different job contains different action, which either modify the state of itself, or modify the game state
    '''
    def reply(self, player: int, exclude: tuple=(), **kwargs) -> Response:
        '''
        The input logic of player
        
        Currently input manually, later it will rewrite by parsing LLM's output
        '''
        clear_screen()

    @abstractmethod
    def execute(self, game: "GameState", player: "Player", **kwargs) -> None:
        '''
        The actual action this job will do
        '''
        pass

    # def update(self, game: "Game", player: "Player", **kwargs) -> None:
    #     '''
    #     An optional update method to update some unique state of the job
    #     '''
    #     pass

    def reset(self) -> None:
        '''
        An optional reset method to reset the state
        '''
        pass

    def vote(self, player: Player, exclude: Sequence) -> int:
        '''
        The vote function

        -1 to give up
        '''
        voted = JobAction.input_an_id('[Vote] Vote whom? ', required=False, exclude=exclude, player_id=player.ID)
        return voted['ID'] if voted else -1
 

class WolfAction(JobAction):

    def __init__(self):
        self.opponents = []

    def provide(self, *args, **kw):
        if 'others' in kw:
            self.opponents = kw.copy()

    def reply(self, player, exclude=(), **_):
        resp = JobAction.input_an_id('[Wolf] Please input an player ID to kill (leave blank to skip): ', False, player.ID, exclude=exclude)
        super().reply(player, exclude)
        return resp
    
    def execute(self, game, _, **kwargs):
        # Pass an ID parameter to kill the player with that ID.
        # The victims are a union, not a tally: every named player dies, so when
        # both wolves name the same player the announcement must not list them
        # twice.
        if 'ID' in kwargs and kwargs['ID'] not in game.night.wolf_kills:
            game.night.wolf_kills.append(kwargs['ID'])


class CivilianAction(JobAction):
    '''
    A civilian does nothing
    '''
    def reply(self, player, exclude=(), **_):
        pass

    def execute(self, game, _, **kwargs): 
        pass


class ProphetAction(JobAction):

    # He need some state to store the result of his skil 
    def __init__(self):
        self.reset()

    def reply(self, player, exclude=(), **_):
        resp = JobAction.input_an_id('[Prophet] Input a player\'s ID to query its state: ', True, player.ID, exclude=exclude)
        super().reply(player, exclude)
        return resp

    def execute(self, game, player, **kwargs):
        if 'ID' in kwargs:
            self.target = kwargs['ID']
            if self.target == player.ID:
                pass # Support operations like this

            self.target_attribute = game.players[self.target].job is Jobs.Wolf

    def reset(self):
        self.target = None
        self.target_attribute = None


class WitchAction(JobAction):

    # The witch is able to see who is killed this night
    def reply(self, player, exclude=(), **kwargs):
        assert 'kills' in kwargs, 'There must be a `kills` params passed'

        kills = kwargs['kills']
        game_print(f'Tonight {(", ".join((f"Player {killed}" for killed in kills))) if kills else "nobody"} died.')
        kill = JobAction.input_an_id('[Witch] Input a player\'s ID to kill: ', False, key='kill', player_id=player.ID, exclude=exclude)
        
        if not kills:
            game_print('Since nobody died tonight, you cannot save anyone')
            save = {}
        else:
            save = JobAction.input_an_id('[Witch] Input a player\'s ID to save (Please select one who was killed tonight): ', False, key='save', player_id=player.ID, exclude=exclude)
        
        super().reply(player, exclude)
        return kill | save

    def execute(self, game: "GameState", _, **kwargs):
        if 'kill' in kwargs:
            game.night.witch_killed_target = kwargs['kill']
        if 'save' in kwargs:
            game.night.witch_saved_target = kwargs['save']        

 
class PlayerState(Enum):
    Alive = 1
    Dead = 2
    Exit = 3


class Camp(Enum):
    Pro = 1
    Wolf = 2

type GameHistory = list[tuple[NightHistory, DayHistory]]


@dataclass
class Player:
    # Every character has a ID and job
    ID: int
    job: Jobs = Jobs.Not_Assigned
    # A flag to label whether the player is alive
    player_state: PlayerState = PlayerState.Alive
    # A player has its own action determined by its job
    job_actions: JobAction | None = None


class GameStage(Enum):
    Initialzing = 1
    Finished_Initializing = 2 
    Night = 3
    Day = 4
    End_Civilian_Win = 5
    End_Wolf_Win = 6
    End_Tie = 7


@dataclass
class NightHistory:
    '''
    Represent game changes at 1 night
    '''
    wolf_kills: list[int] | None = None
    query_target: int | None = None
    witch_killed_target: int | None = None
    witch_saved_target: int | None = None


@dataclass
class DayHistory:
    '''
    Represent game changes at 1 day
    '''
    messages: dict[int, str] | None = None
    # Vote result stores
    # Player: votes & who vote the player
    vote_result: dict[str, tuple[int, list[int]]] | None = None


@dataclass
class GameState:
    # Basic game state (live as long as the whole game process)
    stage: GameStage
    players: dict[int, Player] | None = None
    cycles: int = 1
    exited_players: list[int] | None = None

    # A dict with job as key to query by the game
    job_lookup_table: dict[Jobs, list[int]] | None = None
    # camp_table: dict[Camp, list[int]] | None = None

    # History savings
    history: GameHistory | None = None

    # Daily refreshed state (reset after 1 day-night cycle)
    # wolf_kills: list[int] | None = None
    night: NightHistory | None = None
    day: DayHistory | None = None

def format_player_ids(ids: list[int]) -> str:
    return ", ".join((f'Player {k}' for k in ids))

class Game:

    def __init__(self):
        self.game_state = GameState(GameStage.Initialzing)

    def init_game(self) -> None:
        # Randomly assign job for each player, then tagged by ID
        # Keep the players in a set since we didn't need the order
        
        # Init players and lookup table
        game_print('Welcome to the game!')
        self.game_state.players = {
            i: Player(i) for i in range(6)
        }
        self.game_state.job_lookup_table = { job: [] for job in list(Jobs) if job is not Jobs.Not_Assigned }
        self.game_state.exited_players = []

        # Assign jobs
        remaining = {
            Jobs.Civilian: 2,
            Jobs.Prophet: 1,
            Jobs.Wolf: 2,
            Jobs.Witch: 1
        }

        for job, count in remaining.items():
            # Find candidates who has no job
            candidate_ids = random.sample(sorted(set((k for k, p in self.game_state.players.items() if p.job is Jobs.Not_Assigned))), count)

            # Assign to it
            for i in candidate_ids:
                self.game_state.players[i].job = job
                self.game_state.players[i].job_actions = eval(f'{job.value}Action')()

        # Complete the player_table
        for ID, player in self.game_state.players.items():
            self.game_state.job_lookup_table[player.job].append(ID)

        # Enter the finished stage, game start
        self.game_state.stage = GameStage.Finished_Initializing

    def start(self) -> GameHistory:
        if self.game_state.stage is GameStage.Finished_Initializing:
            self.game_state.stage = GameStage.Night

        # Enter the state machine
        self.proceed()

    def _reset_state_night(self) -> None:
        '''
        Executed if a new night state entered
        '''
        self.game_state.night = NightHistory(wolf_kills=[])

    def _reset_state_day(self) -> None:
        '''
        Executed if a new day state entered
        '''
        self.game_state.day = DayHistory(messages={})

    def _at_night(self) -> None:
        # 1st
        game_print(f'Night {self.game_state.cycles} approach')

        # Only the living may act. A player who has left the game keeps their
        # entry in job_lookup_table, so each job has to be checked for liveness
        # before it is driven, or the dead would keep killing, divining and
        # healing from the grave.
        for wolf_player_id in self.game_state.job_lookup_table[Jobs.Wolf]:
            p = self.game_state.players[wolf_player_id]
            if p.player_state is not PlayerState.Alive:
                continue
            p.job_actions.reset()
            p.job_actions.provide()
            # Avoid killing wolves themselves
            p.job_actions.execute(game.game_state, p, **p.job_actions.reply(player=p, exclude=self.game_state.exited_players + self.game_state.job_lookup_table[Jobs.Wolf]))

        # 2nd
        p_prophet = self.game_state.players[self.game_state.job_lookup_table[Jobs.Prophet][0]]
        if p_prophet.player_state is PlayerState.Alive:
            p_prophet.job_actions.reset()
            p_prophet.job_actions.execute(game.game_state, p_prophet, **p_prophet.job_actions.reply(player=p_prophet, exclude=self.game_state.exited_players + [self.game_state.job_lookup_table[Jobs.Prophet]]))

        # 3rd
        p_witch = self.game_state.players[self.game_state.job_lookup_table[Jobs.Witch][0]]
        if p_witch.player_state is PlayerState.Alive:
            p_witch.job_actions.reset()
            p_witch.job_actions.execute(game.game_state, p_witch, **p_witch.job_actions.reply(player=p_witch, exclude=tuple(set(range(6)) - set(self.game_state.night.wolf_kills)), kills=game.game_state.night.wolf_kills))
        
        # Apply deaths
        for i in self.game_state.night.wolf_kills:
            candidate = self.game_state.players[i]
            if candidate.player_state is not PlayerState.Exit:
                candidate.player_state = PlayerState.Dead

        # Apply save & kill
        if self.game_state.night.witch_killed_target is not None:
            candidate_kill = self.game_state.players[self.game_state.night.witch_killed_target]
            if candidate_kill.player_state is not PlayerState.Exit:
                candidate_kill.player_state = PlayerState.Dead

        if self.game_state.night.witch_saved_target is not None:
            candidate_save = self.game_state.players[self.game_state.night.witch_saved_target]
            if candidate_save.player_state is not PlayerState.Exit:
                candidate_save.player_state = PlayerState.Alive

        self.game_state.stage = GameStage.Day
        self.proceed()

    def _at_day(self) -> None:
        game_print('Night ends and day approches')

        # Announce deaths
        deaths = [k for k, p in self.game_state.players.items() if p.player_state is PlayerState.Dead]
        living = tuple(set(self.game_state.players.keys()) - set(deaths) - set(self.game_state.exited_players))
        if deaths:
            game_print(f'Death(s) at night {self.game_state.cycles}: {format_player_ids(deaths)}')
        else:
            game_print(f'No death at night {self.game_state.cycles}')

        # Update deaths state
        for death in deaths:
            self.game_state.players[death].player_state = PlayerState.Exit
            
        self.game_state.exited_players += deaths

        # Comment
        for i in living:
            self.game_state.players[i].job_actions.comment(self.game_state, self.game_state.players[i])

        game_print('Messages summary:')
        for p, msg in self.game_state.day.messages.items():
            print(f'Player {p}: {msg}')

        # Vote
        game_print('Start voting')
        self.game_state.day.vote_result = { i: [0, []] for i in living }
        for i in living:
            voted = self.game_state.players[i].job_actions.vote(self.game_state.players[i], exclude=self.game_state.exited_players)
            if voted >= 0:
                self.game_state.day.vote_result[voted][0] += 1
                self.game_state.day.vote_result[voted][1].append(i)

        # Check vote result
        result = tuple(v for v, _ in self.game_state.day.vote_result.values())
        if (maxima := max(result)) != min(result):
            candidates = tuple(filter(lambda p: p[1][0] == maxima, self.game_state.day.vote_result.items()))
            candidates_ids = [i for i, _ in candidates]

            # Skip cases with same vote value
            if len(candidates) == 1:
                candidate, votes = candidates[0]
                game_print(f'Player {candidate} out ({votes[0]} votes, by {format_player_ids(votes[1])})')
                self.game_state.exited_players.append(candidate)
                self.game_state.players[candidate].player_state = PlayerState.Exit
            else:
                game_print(f'No one was out. Since candidates {format_player_ids(candidates_ids)} have same votes')

        # Judge whether the game ends
        # The rule is listed below
        '''
        好人胜利：存活狼人数 = 0。
        狼人胜利：满足以下任意一条：
        存活神职数 = 0 (预言家和女巫都死亡)
        存活村民数 = 0 (两个村民都死亡)
        平局：达到最大天数仍未分胜负。
        '''
        civilian_win = all((self.game_state.players[wolf].player_state is not PlayerState.Alive for wolf in self.game_state.job_lookup_table[Jobs.Wolf]))
        if civilian_win:
            game_print('All wolves dead, civilians win!')
            self.game_state.stage = GameStage.End_Civilian_Win
            return

        # The lookup table maps a job to a list of IDs, so the lists have to be
        # flattened before the IDs can be used as player keys
        holy_dead = all(self.game_state.players[k].player_state is not PlayerState.Alive
                        for k in self.game_state.job_lookup_table[Jobs.Prophet] + self.game_state.job_lookup_table[Jobs.Witch])
        civilians_dead = all(self.game_state.players[a].player_state is not PlayerState.Alive
                             for a in self.game_state.job_lookup_table[Jobs.Civilian])
        wolf_win = holy_dead or civilians_dead

        if holy_dead:
            game_print('The prophet and witch were dead')
        if civilians_dead:
            game_print('All civilians are dead')
        if wolf_win:
            game_print('Wolves win!')
            self.game_state.stage = GameStage.End_Wolf_Win
            return

        if self.game_state.cycles > MAXIMUM_CYCLES:
            game_print(f'Game Tie! (more than {MAXIMUM_CYCLES} days passed)')
            self.game_state.stage = GameStage.End_Tie
            return
        
        # Next cycle
        self.game_state.cycles += 1
        self.game_state.stage = GameStage.Night
        game_print('Game continue. Prepare for the next night')

        # Save the history
        self.game_state.history.append((self.game_state.night, self.game_state.day))
        self.proceed()

    def proceed(self) -> None:
        # The logic of the state machine
        match self.game_state.stage:
            case GameStage.Night:
                self._reset_state_night()
                self._at_night()
            case GameStage.Day:
                self._reset_state_day()
                self._at_day()
            case GameStage.End_Tie | GameStage.End_Wolf_Win | GameStage.End_Civilian_Win:
                game_print('Good game! Here\'s a replay for last game:')
                pprint.pprint(self.game_state.history)


if __name__ == '__main__':
    game = Game()
    game.init_game()
    game.start()
