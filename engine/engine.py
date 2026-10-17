import os
import random
import pprint

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

game_print = lambda *args, sep=' ', end='\n': print('[Game]', *args, sep=sep, end=end) 
# Text only the player currently acting may see. The console routes it to a
# private panel that is wiped once that player is done, so it never reaches the
# shared transcript.
game_print_private = lambda *args, sep=' ', end='\n': print('[Game] [Private]', *args, sep=sep, end=end)
game_input = lambda prompt, player: input(f'[Game] [Player {player} operate] {prompt}')


MAXIMUM_CYCLES = 10
PLAYERS_COUNT = 12

def clear_screen():
    if os.system('clear'):
        os.system('cls')

class GameException(Exception):
    pass


class Jobs(Enum):
    '''
    Types of job in our simple game
    '''
    # Also identified by index 0-
    Civilian = 'Civilian'      # 4
    Wolf = 'Wolf'              # 3
    Wolf_King = 'Wolf King'    # 1
    Guard = 'Guard'            # 1
    Witch = 'Witch'            # 1
    Prophet = 'Prophet'        # 1
    Hunter = 'Hunter'          # 1
    Not_Assigned = 'Unknown' 

class OutReason(Enum):
    '''
    How the player died?
    '''
    Wolf_Killed = 1
    Poison_Killed = 2
    Voted = 3
    Identification_Exposed = 4
    Shot_By_Hunter = 5
    Shot_By_Wolf_King = 6
    Not_Out_Yet = 7
    

type Response = dict[str, Any]

class JobAction(ABC):
    '''
    An interface for one job's actions.

    A job overrides only the parts it needs: reply() to ask, execute() to apply
    the answer, on_exit() for a parting shot when it leaves the game. The night
    and day phases drive them in the order GameRule gives.
    '''

    @staticmethod
    def input_an_id(prompt: str, required: bool, player_id: int, key: str = 'ID', exclude: tuple = ()) -> Response:
        ID = ''
        while True: 
            ID_ = game_input(prompt, player_id)
            if not required and not ID_:
                    return {}
            
            try:
                ID = int(ID_)
                assert 0 <= ID < PLAYERS_COUNT and ID not in exclude, 'Invalid player ID'
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

    def comment(self, game: GameState, player: Player, allow_expose: bool = True) -> None:
        '''
        Comment before vote

        `allow_expose` is only meaningful for a wolf: a dead wolf giving 遗言
        speaks without being offered a self-explosion.
        '''
        game.day.messages[player.ID] = game_input('[Addresser] ', player.ID)

    def speech(self, game: GameState, player: Player) -> None:
        '''
        Speech for person who wants to be the police
        '''
        game.day.candidate_speech[player.ID] = game_input(f'[Speecher] ', player.ID)

    def reply(self, player: int, exclude: tuple=(), cycle: int | None = None, **kwargs) -> Response:
        '''
        The input logic of player
        
        Currently input manually, later it will rewrite by parsing LLM's output

        Two pieces of context come from the caller, because the action is only
        asked to shape a question, not to know the world:

        - `exclude` is the set of IDs the player may NOT pick.
        - `cycle` is the night number, for a question that depends on it
          (第一夜不能自救). None means the caller did not say, so night-specific
          rules stay off rather than assuming night 1.
        '''
        clear_screen()

    @abstractmethod
    def execute(self, game: "GameState", player: "Player", cycle: int | None = None, **kwargs) -> None:
        '''
        The actual action this job will do

        `cycle` is here for a job whose *effect* depends on the night; most jobs
        ignore it, and None is safe for them. The player's answer arrives in
        **kwargs, keyed by whatever input_an_id() was told to use.
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

    def on_exit(self, reason: OutReason,  game: "GameState", player: Player) -> None:
        '''
        Special action when a player was exited or voted to be exiled
        '''
        pass

    def vote(self, player: Player, exclude: Sequence) -> int:
        '''
        The vote function

        -1 to give up
        '''
        voted = JobAction.input_an_id('[Vote] Vote whom? ', required=False, exclude=exclude, player_id=player.ID)
        return voted['ID'] if voted else -1

    # We define True -> 上警, False -> 警下, this function is designed like input_an_id, the core of interaction
    # Skip the input fallbacks to False
    def input_yes_or_no(self, prompt: str, game: "GameState", player: Player) -> bool:
        '''
        A yes/no question, for a seat that may answer freely.

        A blank answer means "no": skipping the prompt must not trap the game in
        a re-prompt loop. Unrecognised input is asked again, but only a few
        times - a model that keeps answering something unmappable must not hang
        the game - and then falls back to "no" like a skip does.
        '''
        for _ in range(3):
            answer = game_input(f'{prompt} [y/n] ', player.ID).strip().lower()
            if not answer:
                return False
            if answer in ('y', 'yes', '是', '1'):
                return True
            if answer in ('n', 'no', '否', '0'):
                return False
            game_print('Please answer y or n')
        game_print('Unrecognised answer, treated as no')
        return False
 

class WolfAction(JobAction):

    def __init__(self):
        self.opponents = []

    def provide(self, *args, **kw):
        if 'others' in kw:
            self.opponents = kw.copy()

    def reply(self, player, exclude=(), cycle: int | None = None, **_):
        resp = JobAction.input_an_id('[Wolf] Please input an player ID to kill (leave blank to skip): ', False, player.ID, exclude=exclude)
        super().reply(player, exclude)
        return resp
    
    def execute(self, game, _, cycle: int | None = None, **kwargs):
        # Pass an ID parameter to kill the player with that ID.
        # The victims are a union, not a tally: every named player dies, so when
        # both wolves name the same player the announcement must not list them
        # twice.
        if 'ID' in kwargs and kwargs['ID'] not in game.night.wolf_kills:
            game.night.wolf_kills.append(kwargs['ID'])

    def comment(self, game, player, allow_expose: bool = True) -> bool:
        '''
        Rewrite the method, because the wolf can choose whether to tell their wolf identification
        and then exited

        If >= 2 wolves expose their identification, the police will gone
        '''
        super().comment(game, player, allow_expose)
        if not allow_expose:
            return False
        expose_identification = self.input_yes_or_no('[Choice] Do you want to expose your identification?', game, player)
        return expose_identification


class CivilianAction(JobAction):
    '''
    A civilian does nothing
    '''
    def reply(self, player, exclude=(), cycle: int | None = None, **_):
        pass

    def execute(self, game, _, cycle: int | None = None, **kwargs): 
        pass


class ProphetAction(JobAction):

    # He need some state to store the result of his skil 
    def __init__(self):
        self.reset()

    def reply(self, player, exclude=(), cycle: int | None = None, **_):
        resp = JobAction.input_an_id('[Prophet] Input a player\'s ID to query its state: ', True, player.ID, exclude=exclude)
        super().reply(player, exclude)
        return resp

    def execute(self, game, player, cycle: int | None = None, **kwargs):
        if 'ID' in kwargs:
            self.target = kwargs['ID']
            self.target_attribute = game.players[self.target].job is Jobs.Wolf or game.players[self.target].job is Jobs.Wolf_King

    def reset(self):
        self.target = None
        self.target_attribute = None


class WitchAction(JobAction):
    '''
    The witch holds one poison and one antidote for the whole game.

    She is shown who the pack killed tonight and may spend either - or both - of
    her remaining potions on one of tonight's victims.
    '''

    def __init__(self):
        # Per game, not per night: witch_do_night calls reset() every night, so
        # these must not be initialised there, or they would be refilled nightly.
        self.poison_left = 1
        self.antidote_left = 1

    def antidote_targets(self, kills, player, answer, cycle: int | None = None) -> list[int]:
        '''
        Who the antidote may still rescue tonight.

        These two bars live here rather than in the caller because the caller
        cannot know them: the poison is chosen first, inside reply().

        - 解药救不活被毒死的人: the player she has just poisoned cannot be saved
        - 第一夜不能自救
        '''
        blocked = set()
        if answer.get('kill') is not None:
            blocked.add(answer['kill'])
        if cycle == 1:
            blocked.add(player.ID)
        return [i for i in kills if i not in blocked]

    def reply(self, player, exclude=(), cycle: int | None = None, **kwargs):
        assert 'kills' in kwargs, 'There must be a `kills` params passed'

        kills = kwargs['kills']
        game_print_private(f'Potions left: {self.poison_left} poison, {self.antidote_left} antidote.')
        if kills:
            game_print_private(f'Tonight {", ".join(f"Player {killed}" for killed in kills)} died.')
        else:
            game_print_private('Tonight nobody died.')

        # Only the potions she still holds are offered. A blank answer spends
        # nothing, which is why the counters move in execute() rather than here.
        answer: Response = {}
        if self.poison_left:
            answer |= JobAction.input_an_id('[Witch] Input a player\'s ID to kill: ', False, key='kill', player_id=player.ID, exclude=exclude)
        if self.antidote_left:
            if not kills:
                game_print_private('Since nobody died tonight, you cannot save anyone')
            else:
                # `exclude` above describes the poison's pool (only tonight's
                # victims). The antidote's pool is narrower, and only this turn
                # knows how - see antidote_targets.
                targets = self.antidote_targets(kills, player, answer, cycle)
                if targets:
                    save_exclude = tuple(set(range(PLAYERS_COUNT)) - set(targets))
                    answer |= JobAction.input_an_id('[Witch] Input a player\'s ID to save (Please select one who was killed tonight): ', False, key='save', player_id=player.ID, exclude=save_exclude)
                else:
                    game_print_private('No player you may save tonight.')

        super().reply(player, exclude)
        return answer

    def execute(self, game: "GameState", _, cycle: int | None = None, **kwargs):
        if kwargs.get('kill') is not None:
            game.night.witch_killed_target = kwargs['kill']
            self.poison_left -= 1
        if kwargs.get('save') is not None:
            game.night.witch_saved_target = kwargs['save']
            self.antidote_left -= 1


class WolfKingAction(WolfAction):
    '''
    The Wolf King kills with the pack. His one-off power is not implemented yet.
    '''
    def on_exit(self, reason, game, player):
        if reason is OutReason.Voted:
            target_player_id = self.input_an_id('You can shoot 1 person, input an ID: ', False, player.ID, exclude=tuple(set(game.exited_players) | set([player.ID])))
            if 'ID' in target_player_id:
                game_print(f'Player {target_player_id["ID"]} was shot by player {player.ID}')
                if GameRuleDefinitions.should_transmit_the_budge(game, target_player_id['ID']):
                    GameRuleDefinitions.police_badge_transmission(game)

                # the id, not the answer dict (same fix as the hunter's shot)
                GameRuleDefinitions.exit_player(game, target_player_id['ID'], OutReason.Shot_By_Wolf_King)
            else:
                game_print('No one was shot')


class GuardAction(JobAction):
    '''
    The guard protects one player a night, at most once each.

    reply() must return the answer under the key execute() reads, or the choice
    is silently dropped.
    '''
    def reply(self, player, exclude=(), cycle: int | None = None, **_):
        resp = JobAction.input_an_id('[Guard] Input a player\'s ID to guard (No double guarding): ', False, player.ID, key='target', exclude=exclude)
        super().reply(player, exclude)
        return resp

    def execute(self, game: "GameState", _, cycle: int | None = None, **kwargs):
        if kwargs.get('target') is None:
            return
        # The defence in depth: reply() already refuses a repeat, but this holds
        # even if a caller passes a stale exclude. Recording the pick is
        # end_night's job, so there is one writer for guarded_players.
        if kwargs['target'] not in (game.guarded_players or []):
            game.night.guard_target = kwargs['target']


class HunterAction(JobAction):
    '''
    Placeholder: the hunter's dying shot is not implemented yet.
    '''
    def reply(self, player, exclude=(), cycle: int | None = None, **_):
        return {}

    def execute(self, game, _, cycle: int | None = None, **kwargs):
        pass

    def on_exit(self, reason, game, player):
        if reason is OutReason.Voted or reason is OutReason.Wolf_Killed:
            target_player_id = self.input_an_id('You can shoot 1 person, input an ID: ', False, player.ID, exclude=tuple(set(game.exited_players) | set([player.ID])))
            if 'ID' in target_player_id:
                game_print(f'Player {target_player_id["ID"]} was shot by player {player.ID}')
                if GameRuleDefinitions.should_transmit_the_budge(game, target_player_id['ID']):
                    GameRuleDefinitions.police_badge_transmission(game)
                # the id, not the answer dict: this line used to pass the whole
                # {'ID': n} mapping and raised TypeError on every real shot
                GameRuleDefinitions.exit_player(game, target_player_id['ID'], OutReason.Shot_By_Hunter) 
            else:
                game_print('No one was shot')


# Which class implements which job. Adding a job means adding an entry here,
# rather than deriving a class name from the job's display value.
JOB_ACTIONS: dict[Jobs, type[JobAction]] = {
    Jobs.Civilian: CivilianAction,
    Jobs.Wolf: WolfAction,
    Jobs.Wolf_King: WolfKingAction,
    Jobs.Guard: GuardAction,
    Jobs.Witch: WitchAction,
    Jobs.Prophet: ProphetAction,
    Jobs.Hunter: HunterAction,
}


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

    # How was the player out
    out_reason: OutReason = OutReason.Not_Out_Yet


class GameStage(Enum):
    Initializing = 1
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
    guard_target: int | None = None


@dataclass
class DayHistory:
    '''
    Represent game changes at 1 day
    '''
    messages: dict[int, str] | None = None
    # Vote result stores
    # Player: votes & who vote the player
    vote_result: dict[str, tuple[int, list[int]]] | None = None

    # Deaths' comment on the first day
    deaths_comment: list[str] | None = None

    # This is a list will be cleared, at the end of the day, if this is not empty, it will be appended to the main game state
    wolf_exposed_this_day: list[str] | None = None

    # Police voting result (None if not exist)
    # 上警 / 警下状态 (ID -> camp)
    police_vote_person_camp: dict[bool, list[int]] | None = None
    # 上警的发言状态 (ID -> speech)
    candidate_speech: dict[int, str] | None = None
    # 警下的投票状态 (Who vote whom)
    police_voting_result: dict[int, int] | None = None
    # 最终决定的警长
    the_police: int | None = None
    # 是否出现撕毁警徽, 值为 None 即为非警徽流 
    break_the_police: bool | None = None


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

    # Players who have already been guarded once 
    guarded_players: list[int] | None = None

    # The wolf(ves) who exposed their identification
    wolf_exposed: list[int] | None = None

    # History savings
    history: GameHistory | None = None

    # Daily refreshed state (reset after 1 day-night cycle)
    # wolf_kills: list[int] | None = None
    night: NightHistory | None = None
    day: DayHistory | None = None

def format_player_ids(ids: list[int]) -> str:
    return ", ".join((f'Player {k}' for k in ids))


class GameRule:
    '''
    The actual configuration for the game rule

    Everything that separates one ruleset from another lives here, so the phase
    definitions below stay generic and a new job is mostly a matter of data.
    '''
    Version = '2.0'

    # How many of each job a full table holds. Must add up to PLAYERS_COUNT.
    Roles = {
        Jobs.Civilian: 4,
        Jobs.Prophet: 1,
        Jobs.Wolf: 3,
        Jobs.Wolf_King: 1,
        Jobs.Witch: 1,
        Jobs.Guard: 1,
        Jobs.Hunter: 1
    }

    # The jobs making up each side, used by the winning conditions
    WolfCamp = (Jobs.Wolf, Jobs.Wolf_King)
    GodRoles = (Jobs.Prophet, Jobs.Witch, Jobs.Guard, Jobs.Hunter)
    CivilianRoles = (Jobs.Civilian,)

    # Phase names, run in this order as f'{phase}_night' / f'{phase}_day'
    Night = [
        'guard_do', 'wolf_do', 'prophet_do', 'witch_do'
    ]

    Day = [
        'comment', 'vote_and_exile', 'determine_game_process'
    ]


class GameRuleDefinitions:
    '''
    Defining the rules

    One static method per phase, taking the GameState it mutates. GameRule names
    the phases and their order; Game._at_night/_at_day run them.

    A phase that is not implemented yet is left as `pass`: it still runs in its
    place in the order, it just has no effect.
    '''

    @staticmethod
    def players_with(state: "GameState", roles: Sequence[Jobs]) -> list[int]:
        '''Every ID holding one of `roles`, in seat order.'''
        return [player_id for role in roles for player_id in state.job_lookup_table[role]]

    @staticmethod
    def any_alive(state: "GameState", ids: Sequence[int]) -> bool:
        return any(state.players[player_id].player_state is PlayerState.Alive for player_id in ids)

    @staticmethod
    def living_ids(state: "GameState") -> list[int]:
        '''
        Everyone still in the game.

        begin_day retires the night's dead, so this is simply "has not exited".
        '''
        return [player_id for player_id, player in state.players.items()
                if player.player_state is not PlayerState.Exit]

    @staticmethod
    def death_reason(state: "GameState", player_id: int) -> OutReason:
        '''
        How a player died tonight. The witch's poison outranks the pack.
        Should be only used at the begin of day (since other reason like voted won't be considered in this method)
        '''
        if player_id == state.night.witch_killed_target:
            return OutReason.Poison_Killed
        return OutReason.Wolf_Killed

    @staticmethod
    def exit_player(state: "GameState", player_id: int,
                    reason: OutReason = OutReason.Not_Out_Yet) -> None:
        '''
        Take a player out of the game, in one place.

        Every route out - the night's deaths at dawn and the day's vote - comes
        through here, so the reason is recorded and the player's own parting
        action runs exactly once.

        Dead is not the same as Exit: a night death is only *marked* dead until
        the dawn announcement, because the witch's potion can still save it.
        '''
        player = state.players[player_id]
        if player.player_state is PlayerState.Exit:
            return
        player.player_state = PlayerState.Exit
        player.out_reason = reason
        if player_id not in (state.exited_players or []):
            state.exited_players.append(player_id)
        player.job_actions.on_exit(reason, state, player)

    @staticmethod
    def begin_night(state: "GameState") -> None:
        state.night = NightHistory(wolf_kills=[])
        game_print(f'Night {state.cycles} approach')

    @staticmethod
    def wolf_do_night(state: "GameState") -> None:
        '''
        Every living wolf names a victim, and a wolf may pass.

        The Wolf King is a wolf, so he hunts with the pack; his one-off power is
        a rule of its own and is not implemented yet.
        '''
        
        pack = GameRuleDefinitions.players_with(state, GameRule.WolfCamp)
        for player_id in pack:
            p = state.players[player_id]
            if p.player_state is not PlayerState.Alive:
                continue
            p.job_actions.reset()
            p.job_actions.provide()
            # Avoid killing wolves themselves (Now this rule removed in 12-seat game rule)
            p.job_actions.execute(state, p, **p.job_actions.reply(
                player=p, cycle=state.cycles, exclude=tuple(state.exited_players or ())))

    @staticmethod
    def prophet_do_night(state: "GameState") -> None:
        for player_id in state.job_lookup_table[Jobs.Prophet]:
            p = state.players[player_id]
            if p.player_state is not PlayerState.Alive:
                continue
            p.job_actions.reset()
            p.job_actions.execute(state, p, **p.job_actions.reply(
                player=p, cycle=state.cycles, exclude=tuple(state.exited_players or ()) + (player_id,)))

    @staticmethod
    def witch_do_night(state: "GameState") -> None:
        '''
        The witch is shown tonight's victims and may poison or save one of them.
        '''
        tonight = state.night.wolf_kills or []
        for player_id in state.job_lookup_table[Jobs.Witch]:
            p = state.players[player_id]
            if p.player_state is not PlayerState.Alive:
                continue
            p.job_actions.reset()
            p.job_actions.execute(state, p, **p.job_actions.reply(
                player=p,
                # 毒药: 允许对任意非己存活目标使用, so the poison's pool is every
                # living seat but her own. The antidote's pool is different -
                # tonight's victims only - and is worked out inside reply(), where
                # the poison choice is already known.
                exclude=tuple(set(state.exited_players or ()) | {player_id}),
                cycle=state.cycles,
                kills=tonight))

    @staticmethod
    def civilian_do_night(state: "GameState") -> None:
        '''A civilian holds no night action.'''
        pass

    @staticmethod
    def hunter_do_night(state: "GameState") -> None:
        '''
        A hunter did nothing at night
        '''
        pass

    @staticmethod
    def guard_do_night(state: "GameState") -> None:
        '''
        Note only 1 guard in our game, for-loop for better compatability
        '''
        for player_id in (guards := state.job_lookup_table[Jobs.Guard]):
            p = state.players[player_id]
            if p.player_state is not PlayerState.Alive:
                continue
            p.job_actions.reset()
            p.job_actions.execute(state, p, **p.job_actions.reply(
                player=p,
                cycle=state.cycles,
                # `exclude` is the *disallowed* set - input_an_id rejects any ID
                # that is in it - so name the two groups the guard may not pick:
                # the out of play, and whoever he protected last night. He may
                # pick himself: 可以守护自己, 只是不能连守.
                exclude=tuple(set(state.exited_players or ()) | set(state.guarded_players or []))))


    @staticmethod
    def end_night(state: "GameState") -> None:
        '''Apply the night's deaths and the witch's potion, then break the dawn.'''
        # 特殊规则如下:
        '''
        女巫先毒药再解药 (因为女巫的毒药不能被守护, 也不能救毒药毒的人)
        同守同救判定死亡
        '''

        # Round I (Wolves killing effect)
        for i in state.night.wolf_kills or ():
            candidate = state.players[i]
            # Add a extra check for guard's protection
            if candidate.player_state is not PlayerState.Exit and i != state.night.guard_target:
                candidate.player_state = PlayerState.Dead

        # Round II (Save whom was killed)
        if state.night.witch_saved_target is not None:
            saved = state.night.witch_saved_target
            candidate_save = state.players[saved]
            if candidate_save.player_state is not PlayerState.Exit:
                # 同守同救: guarding and saving the same player kills them, so the
                # potion must not be allowed to undo it
                if saved == state.night.guard_target:
                    candidate_save.player_state = PlayerState.Dead
                else:
                    candidate_save.player_state = PlayerState.Alive

        # Round III (Use poison to kill person)
        if state.night.witch_killed_target is not None:
            candidate_kill = state.players[state.night.witch_killed_target]
            if candidate_kill.player_state is not PlayerState.Exit:
                candidate_kill.player_state = PlayerState.Dead

        # Remember tonight's protection so the next night can forbid repeating it
        # (不能连守). Assign rather than append: the rule is about consecutive
        # nights, so only the most recent pick matters. Without this the exclude
        # in guard_do_night and the check in GuardAction.execute both read an
        # empty list, and the same player can be guarded every night.
        state.guarded_players = ([state.night.guard_target]
                                 if state.night.guard_target is not None else [])

        state.stage = GameStage.Day

    @staticmethod
    def vote_police(state: "GameState") -> None:
        '''
        Happenes only at day 1
        '''
        # Determine camp
        game_print('Now we can select a police. You can select whether you will be an candidate')
        for player_id in GameRuleDefinitions.living_ids(state):
            p = state.players[player_id]
            r = p.job_actions.input_yes_or_no('[Option] Choose to be the police?', state, p)
            state.day.police_vote_person_camp[r].append(player_id)

        # Speech
        if state.day.police_vote_person_camp[True]:
            game_print('Next is the candidites\' speech')
            for up_player_id in state.day.police_vote_person_camp[True]:
                p = state.players[up_player_id]
                p.job_actions.speech(state, p)

            game_print('Speech summary')
            for k, v in state.day.candidate_speech.items():
                game_print(f'Player {k} want to be the police: {v}')
        else:
            game_print('No one choose to be the police, skipped')

        # Vote (if and only if we have two or more candidates)
        if len(state.day.police_vote_person_camp[True]) > 1:
            game_print('Now we\'re going to vote a police')
            state.day.police_voting_result = { i: 0 for i in state.day.police_vote_person_camp[True] }
            for down_player_id in state.day.police_vote_person_camp[False]:
                p = state.players[down_player_id]
                votes = p.job_actions.vote(p, exclude=tuple(set(range(PLAYERS_COUNT)) - set(state.day.police_vote_person_camp[True])))
                # -1 is a legal abstention (vote() documents it), so it is not a tally entry
                if votes >= 0:
                    state.day.police_voting_result[votes] += 1

            # Calculate votes. The values are plain counts, so compare them
            # directly rather than indexing them like the exile tally's lists.
            counts = tuple(state.day.police_voting_result.values())
            maxima = max(counts)
            candidates_ids = [i for i, count in state.day.police_voting_result.items() if count == maxima]

            if len(candidates_ids) == 1:
                state.day.the_police = candidates_ids[0]
                state.day.break_the_police = False
            else:
                # 平票则本局无警长
                game_print(f'No police for this game, {format_player_ids(candidates_ids)} have the same vote')
                state.day.the_police = None
        elif len(state.day.police_vote_person_camp[True]) == 1:
            game_print(f'Since only Player {(p_id := state.day.police_vote_person_camp[True][0])} choose to be the candidate, the player will be the police')
            state.day.the_police = p_id
            state.day.break_the_police = False

    @staticmethod
    def police_badge_transmission(state: "GameState") -> None:
        game_print(f'The police (Player {state.day.the_police}) is dead')
        police = state.players[state.day.the_police]
        
        # Note True -> transmit, False -> break
        transmit_or_break = police.job_actions.input_yes_or_no('Do you want to transmit your badge to a new player?', state, police)
        if transmit_or_break:
            # Only a living player may take the badge. Testing Alive rather than
            # "not exited" matters: the night's other victims are still Dead at
            # this moment, so a corpse would otherwise be an acceptable target.
            candidates = [pid for pid, p in state.players.items()
                          if p.player_state is PlayerState.Alive and pid != state.day.the_police]
            if not candidates:
                game_print('No living player can take the badge, so it is torn up')
                state.day.break_the_police = True
                state.day.the_police = None
                return
            new_police = police.job_actions.input_an_id(
                'Which player do you want to send your badge?', required=False, player_id=police.ID, 
                exclude=tuple(set(range(PLAYERS_COUNT)) - set(candidates)) # The police can be moved to any living player except himself
            )
            if 'ID' not in new_police:
                # Skipping the hand-over tears the badge, the same way a blank
                # answer means "no" everywhere else. Nothing can then loop.
                game_print('The badge was not handed over, so it is torn up')
                state.day.break_the_police = True
                state.day.the_police = None
                return
            state.day.the_police = new_police['ID']
        else:
            # After the break, there's no police
            state.day.break_the_police = True
            state.day.the_police = None

    @staticmethod
    def should_transmit_the_budge(state: "GameState", target_dead: int) -> bool:
        return state.day.the_police is not None and (not state.day.break_the_police) and state.day.the_police == target_dead

    @staticmethod
    def begin_day(state: "GameState") -> None:
        '''Announce the night's deaths and retire them.'''
        # On the very first dawn there is no previous day yet, so the badge
        # state has to be read defensively.
        previous = state.day
        state.day = DayHistory(
            messages={},
            police_vote_person_camp={ True: [], False: [] },
            candidate_speech={},
            # Initialize a place to store the deaths' comment on the first day,
            deaths_comment=[] if state.cycles == 1 else None,
            wolf_exposed_this_day=[],
            # Don't forget to pass the police badge state
            the_police=previous.the_police if previous is not None else None,
            break_the_police=previous.break_the_police if previous is not None else None
        )
        game_print('Night ends and day approches')

        deaths = [player_id for player_id, player in state.players.items()
                  if player.player_state is PlayerState.Dead]

        # No explicit hunter call here: exit_player runs on_exit exactly once,
        # with the true cause (death_reason), so a hunter killed by the wolves
        # shoots once and a poisoned hunter does not shoot at all.

        # Apply the real deaths
        for death in deaths:
            if GameRuleDefinitions.should_transmit_the_budge(state, death):
                # Possible police budge transmission here
                GameRuleDefinitions.police_badge_transmission(state)
            GameRuleDefinitions.exit_player(state, death,
                                           GameRuleDefinitions.death_reason(state, death))

        if deaths:
            game_print(f'Death(s) at night {state.cycles}: {format_player_ids(deaths)}')
        else:
            game_print(f'No death at night {state.cycles}')

        # For the first day, we need to vote the police
        if state.cycles == 1:
            GameRuleDefinitions.vote_police(state)

        # Check whether the game ends here (first time)
        if GameRuleDefinitions.determine_game_process_day(state):
            # Maybe sending some extra notice here
            return

    @staticmethod
    def comment_day(state: "GameState") -> None:
        '''Every living player makes one public statement.'''

        # The first day we have chance for everyone to comment
        # This should be public
        if state.cycles == 1:
            GameRuleDefinitions.deaths_comment_day(state)

        # Note the police can determine the order of comment (For simplicity, start from the police, clockwise / counterclockwise)
        livings = GameRuleDefinitions.living_ids(state)
        comment_order = []
        if state.day.the_police is not None:
            police = state.players[state.day.the_police]
            comment_order.append(state.day.the_police)
            livings_without_police = list(set(livings) - {state.day.the_police})
            clockwise = police.job_actions.input_yes_or_no(
                'Determine the comment order, start from you, yes for clockwise, no for counterclockwise',
                state, police)
            livings_without_police.sort(reverse=clockwise)
            comment_order += livings_without_police
            game_print(f'The police has determined the comment order: {"->".join((f"Player {p}" for p in comment_order))}')
        else:
            comment_order = livings
            game_print('Since no police, the comment order fallbacks to default order')

        # The comment will shared among every players in real-time
        for i in comment_order:
            # Check the player's job
            # If it is wolf, check whether the wolf expose his identification
            if i in state.job_lookup_table[Jobs.Wolf] + state.job_lookup_table[Jobs.Wolf_King]:
                exposed = state.players[i].job_actions.comment(state, state.players[i])
                if exposed:
                    state.day.wolf_exposed_this_day.append(i)
                    # Kill and exit instantly
                    GameRuleDefinitions.exit_player(state, i, OutReason.Identification_Exposed)
                    # 警徽在警长死亡后移交或撕毁 - 自爆也是出局, so the badge cannot
                    # stay on a player who has just left the game.
                    if GameRuleDefinitions.should_transmit_the_budge(state, i):
                        GameRuleDefinitions.police_badge_transmission(state)
                    if GameRuleDefinitions.determine_game_process_day(state):
                        return
                    # 自他自爆完后直接跳过后续玩家的发言; the vote is skipped by the
                    # block after this loop, which switches the stage to Night
                    break
            else:
                state.players[i].job_actions.comment(state, state.players[i])

        game_print('Messages summary:')
        for p, msg in state.day.messages.items():
            game_print(f'Player {p}: {msg}')

        # The rule, once there's a wolf tell his identification, we break the current day process and enter next night instantly
        if state.day.wolf_exposed_this_day:
            game_print('There exists wolf who tell his identification, skip the vote and exile process, next night approaches')
            state.wolf_exposed += state.day.wolf_exposed_this_day

            # >= 2 wolves exposure
            if len(state.wolf_exposed) >= 2:
                # Remove the badge
                state.day.break_the_police = True
                state.day.the_police = None

            state.cycles += 1
            state.stage = GameStage.Night
            state.history.append((state.night, state.day))

    @staticmethod
    def deaths_comment_day(state: "GameState") -> None:
        '''
        Happened only on the 1st day
        '''
        game_print('It\'s time for deaths to speak before they are out')
        for death in state.exited_players:
            # 遗言 is a plain statement: a wolf who is already out cannot self-explode
            state.players[death].job_actions.comment(state, state.players[death], allow_expose=False)

    @staticmethod
    def vote_and_exile_day(state: "GameState") -> None:
        '''Everyone votes; the largest tally leaves, a tie means nobody leaves.'''
        game_print('Start voting')
        living = GameRuleDefinitions.living_ids(state)
        state.day.vote_result = { i: [0, []] for i in living }
        for i in living:
            voted = state.players[i].job_actions.vote(state.players[i], exclude=state.exited_players)

            vote_value = 1
            if state.day.the_police is not None and (not state.day.break_the_police) and state.day.the_police == i:
                vote_value += 0.5

            if voted >= 0:
                state.day.vote_result[voted][0] += vote_value
                state.day.vote_result[voted][1].append(i)

        # Check vote result
        result = tuple(v for v, _ in state.day.vote_result.values())
        if (maxima := max(result)) != min(result):
            candidates = tuple(filter(lambda p: p[1][0] == maxima, state.day.vote_result.items()))
            candidates_ids = [i for i, _ in candidates]

            # 放逐时 注意猎人 | 狼王的技能效果, 同时注意警长的警徽转让, 并且做狼人自爆的警徽检查
            # 猎人被刀/放逐可以开枪, 狼王仅能在被放逐时可以开枪
            # The hunter's skill and wolf king's skill makes difference here
            # Skip cases with same vote value
            if len(candidates) == 1:
                candidate, votes = candidates[0]
                # The hunter's and wolf king's parting shots come from
                # exit_player's on_exit hook; calling it here as well would let
                # them shoot twice.
                game_print(f'Player {candidate} out ({votes[0]} votes, by {format_player_ids(votes[1])})')
                if GameRuleDefinitions.should_transmit_the_budge(state, candidate):
                    GameRuleDefinitions.police_badge_transmission(state)

                GameRuleDefinitions.exit_player(state, candidate, OutReason.Voted)
            else:
                game_print(f'No one was out. Since candidates {format_player_ids(candidates_ids)} have same votes')
        else:
            game_print(f'No one was out. Since everyone who was voted has the same votes')

    @staticmethod
    def determine_game_process_day(state: "GameState") -> bool:
        '''
        Decide whether the game is over, and set the final stage if it is.

        好人胜利：存活狼人数 = 0。
        狼人胜利：存活神职数 = 0（预言家、女巫、守卫和猎人都死亡），
                  或存活村民数 = 0。
        平局：达到最大天数仍未分胜负。

        Both sides are counted from GameRule's camp tables. A side with no
        members at all cannot satisfy its own condition: all() over an empty
        sequence is True, which would end the game on the very first day.

        Note: Returns True if game ends in any result, False for game continuing
        '''
        wolves = GameRuleDefinitions.players_with(state, GameRule.WolfCamp)
        gods = GameRuleDefinitions.players_with(state, GameRule.GodRoles)
        civilians = GameRuleDefinitions.players_with(state, GameRule.CivilianRoles)

        if wolves and not GameRuleDefinitions.any_alive(state, wolves):
            game_print('All wolves dead, civilians win!')
            state.stage = GameStage.End_Civilian_Win
            return True

        gods_dead = bool(gods) and not GameRuleDefinitions.any_alive(state, gods)
        civilians_dead = bool(civilians) and not GameRuleDefinitions.any_alive(state, civilians)

        if gods_dead:
            game_print('All gods are dead')
        if civilians_dead:
            game_print('All civilians are dead')
        if gods_dead or civilians_dead:
            game_print('Wolves win!')
            state.stage = GameStage.End_Wolf_Win
            return True

        if state.cycles > MAXIMUM_CYCLES:
            game_print(f'Game Tie! (more than {MAXIMUM_CYCLES} days passed)')
            state.stage = GameStage.End_Tie
            return True

        return False

    @staticmethod
    def end_day(state: "GameState") -> None:
        '''
        If the day did not end the game, count the cycle and go back to night.

        determine_game_process_day sets an End_* stage when the game is over, in
        which case there is no next cycle.

        Abnormal process (like there's a wolf expose his identification, won't run this)
        '''
        if state.stage is not GameStage.Day:
            return

        state.cycles += 1
        state.stage = GameStage.Night
        game_print('Game continue. Prepare for the next night')

        # Save the history
        state.history.append((state.night, state.day))


class Game:

    def __init__(self):
        self.game_state = GameState(GameStage.Initializing)

    def init_game(self) -> None:
        # Randomly assign job for each player, then tagged by ID
        # Keep the players in a set since we didn't need the order
        
        # Init players and lookup table
        game_print('Welcome to the game!')
        self.game_state.players = {
            i: Player(i) for i in range(PLAYERS_COUNT)
        }
        self.game_state.job_lookup_table = { job: [] for job in list(Jobs) if job is not Jobs.Not_Assigned }
        self.game_state.exited_players = []
        self.game_state.guarded_players = []
        self.game_state.history = []
        self.game_state.wolf_exposed = []

        # Assign jobs, as the rule configuration describes them
        seated = sum(GameRule.Roles.values())
        if seated != PLAYERS_COUNT:
            raise GameException(
                f'GameRule.Roles describes {seated} players but PLAYERS_COUNT is {PLAYERS_COUNT}')

        for job, count in GameRule.Roles.items():
            # Find candidates who has no job
            candidate_ids = random.sample(sorted(set((k for k, p in self.game_state.players.items() if p.job is Jobs.Not_Assigned))), count)

            # Assign to it
            for i in candidate_ids:
                self.game_state.players[i].job = job
                self.game_state.players[i].job_actions = JOB_ACTIONS[job]()

        # Complete the player_table
        for ID, player in self.game_state.players.items():
            self.game_state.job_lookup_table[player.job].append(ID)

        # Enter the finished stage, game start
        self.game_state.stage = GameStage.Finished_Initializing

    def start(self) -> None:
        if self.game_state.stage is GameStage.Finished_Initializing:
            self.game_state.stage = GameStage.Night

        # Enter the state machine
        self.proceed()

    def _at_night(self) -> None:
        '''Drive one night: GameRule names the phases and their order.'''
        GameRuleDefinitions.begin_night(self.game_state)
        for phase in GameRule.Night:
            # Announced in the same words whether or not the job still has a
            # living holder. The envelope is the only night-time trace an
            # observer gets, so anything that varied with liveness - a skipped
            # phase, a "nobody can act" note - would give the roster away.
            game_print(f'[{phase}] begins')
            getattr(GameRuleDefinitions, f'{phase}_night')(self.game_state)
            game_print(f'[{phase}] ends')
        GameRuleDefinitions.end_night(self.game_state)
        self.proceed()

    def _at_day(self) -> None:
        '''Drive one day: GameRule names the phases and their order.'''
        GameRuleDefinitions.begin_day(self.game_state)
        for phase in GameRule.Day:
            if self.game_state.stage is GameStage.Day:
                # We have some cases that will skip the day process
                getattr(GameRuleDefinitions, f'{phase}_day')(self.game_state)
            else:
                break

        if self.game_state.stage is GameStage.Day:
            GameRuleDefinitions.end_day(self.game_state)
        self.proceed()

    def proceed(self) -> None:
        # The logic of the state machine
        match self.game_state.stage:
            case GameStage.Night:
                self._at_night()
            case GameStage.Day:
                self._at_day()
            case GameStage.End_Tie | GameStage.End_Wolf_Win | GameStage.End_Civilian_Win:
                game_print(f'Good game! The game finally ends, lasting {self.game_state.cycles} cycles')
                # pprint.pprint(self.game_state.history)


if __name__ == '__main__':
    game = Game()
    game.init_game()
    game.start()
