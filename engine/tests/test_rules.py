'''
One test per rule in rule.md.

Fixture tests pin a single rule; game tests check that whole games respect it.
'''
from __future__ import annotations

import random
import unittest

import support
import engine

def vote_to(target: int, shoot: int | None = None):
    '''A pick() that exiles `target` and, if asked, shoots `shoot`.'''
    def choose(key, legal, voter, prompt):
        if 'shoot' in prompt:
            return shoot if shoot is not None else random.choice(legal)
        if 'Vote whom' in prompt:
            return target
        return random.choice(legal)
    return choose


Jobs = engine.Jobs
PlayerState = engine.PlayerState
OutReason = engine.OutReason


class Roles(unittest.TestCase):
    def test_table_matches_the_rule(self):
        expected = {Jobs.Civilian: 4, Jobs.Wolf: 3, Jobs.Wolf_King: 1,
                    Jobs.Witch: 1, Jobs.Prophet: 1, Jobs.Guard: 1, Jobs.Hunter: 1}
        self.assertEqual(engine.GameRule.Roles, expected)
        self.assertEqual(sum(engine.GameRule.Roles.values()), engine.PLAYERS_COUNT)

    def test_camps(self):
        self.assertEqual(set(engine.GameRule.WolfCamp), {Jobs.Wolf, Jobs.Wolf_King})
        self.assertEqual(set(engine.GameRule.GodRoles),
                         {Jobs.Prophet, Jobs.Witch, Jobs.Guard, Jobs.Hunter})
        self.assertEqual(set(engine.GameRule.CivilianRoles), {Jobs.Civilian})

    def test_night_order_follows_the_rule(self):
        # 守卫 -> 狼 -> 预言家 -> 女巫
        self.assertEqual(engine.GameRule.Night,
                         ['guard_do', 'wolf_do', 'prophet_do', 'witch_do'])

    def test_every_job_has_an_action_class(self):
        for job in Jobs:
            if job is Jobs.Not_Assigned:
                continue
            self.assertIn(job, engine.JOB_ACTIONS, job)


class WolfPack(unittest.TestCase):
    '''共同商议: 3 狼 + 1 狼王 商定至多一名受害者.'''

    def build(self, dead=()):
        state = support.new_state(jobs={1: Jobs.Wolf, 2: Jobs.Wolf, 3: Jobs.Wolf_King})
        for player_id in dead:
            support.kill(state, player_id)
        return state

    def run_pack(self, state, picks, pass_rate=0.0):
        with support.Driver(pass_rate=pass_rate) as driver:
            driver.pick = lambda key, legal, voter, prompt: picks[voter]
            engine.GameRuleDefinitions.wolf_do_night(state)
        return driver

    def test_the_most_proposed_player_is_the_only_victim(self):
        state = self.build()
        self.run_pack(state, {1: 5, 2: 5, 3: 6})
        self.assertEqual(state.night.wolf_kills, [5],
                         'the pack agreed on one player, so only that player dies')

    def test_a_split_pack_kills_nobody(self):
        state = self.build()
        self.run_pack(state, {1: 5, 2: 6, 3: 7})
        self.assertEqual(state.night.wolf_kills, [],
                         'a tie is not agreement: 平票则无人死亡')

    def test_a_pack_that_all_passes_kills_nobody(self):
        state = self.build()
        self.run_pack(state, {1: 5, 2: 6, 3: 7}, pass_rate=1.0)
        self.assertEqual(state.night.wolf_kills, [])

    def test_every_living_wolf_proposes_including_the_king(self):
        state = self.build(dead=[2])
        driver = self.run_pack(state, {1: 5, 3: 5})
        self.assertEqual(sorted(state.night.wolf_proposals), [1, 3],
                         'a dead wolf does not take part, the Wolf King does')
        self.assertEqual([player for _, player, _, _ in driver.prompts], [1, 3],
                         'the pack speaks in seat order')

    def test_the_pack_may_propose_its_own(self):
        state = self.build()
        driver = self.run_pack(state, {1: 3, 2: 3, 3: 3})
        self.assertIn(1, driver.legal['ID'])
        self.assertIn(3, driver.legal['ID'])
        self.assertEqual(state.night.wolf_kills, [3], '可以杀同阵营的狼')

    def test_an_exited_seat_is_not_a_target(self):
        state = self.build()
        support.exited(state, 4)
        driver = self.run_pack(state, {1: 5, 2: 5, 3: 5})
        self.assertNotIn(4, driver.legal['ID'])


class Prophet(unittest.TestCase):
    def test_sees_the_wolf_king_as_a_wolf(self):
        state = support.new_state(jobs={1: Jobs.Prophet, 3: Jobs.Wolf_King, 4: Jobs.Wolf})
        action = state.players[1].job_actions
        action.execute(state, state.players[1], ID=3)
        self.assertTrue(action.target_attribute, 'the Wolf King must read as 狼阵营')
        action.execute(state, state.players[1], ID=4)
        self.assertTrue(action.target_attribute)
        action.execute(state, state.players[1], ID=2)
        self.assertFalse(action.target_attribute)

    def test_cannot_target_herself_or_the_out(self):
        state = support.new_state(jobs={1: Jobs.Prophet})
        support.exited(state, 5)
        with support.Driver() as driver:
            driver.pending['v'] = '2'
            engine.GameRuleDefinitions.prophet_do_night(state)
        self.assertNotIn(1, driver.legal['ID'], 'she must not be able to query herself')
        self.assertNotIn(5, driver.legal['ID'], 'an exited player is not a target')


class Witch(unittest.TestCase):
    def setUp(self):
        self.state = support.new_state(jobs={0: Jobs.Witch})
        self.state.night.wolf_kills = [5]

    def run_witch(self, cycle=None):
        self.state.cycles = cycle or self.state.cycles
        with support.Driver() as driver:
            engine.GameRuleDefinitions.witch_do_night(self.state)
        return driver

    def test_poison_may_target_any_living_player_but_herself(self):
        support.exited(self.state, 3)
        driver = self.run_witch()
        legal = driver.legal['kill']
        self.assertNotIn(0, legal, '毒药 cannot target herself')
        self.assertNotIn(3, legal, 'nor an exited player')
        self.assertIn(7, legal, '任意非己存活目标')

    def test_antidote_only_targets_tonights_victims(self):
        driver = self.run_witch()
        self.assertEqual(driver.legal['save'], [5])

    def test_antidote_excludes_the_player_she_poisons(self):
        with support.Driver(pass_rate=0.0) as driver:
            # poison the only victim, which leaves the antidote with no target
            driver.pick = lambda key, legal, voter, prompt: 5
            engine.GameRuleDefinitions.witch_do_night(self.state)
        self.assertNotIn('save', driver.legal,
                         '解药救不活被毒死的人, so the prompt is not even offered')

    def test_first_night_cannot_save_herself(self):
        self.state.night.wolf_kills = [0]
        driver = self.run_witch(cycle=1)
        self.assertNotIn('save', driver.legal, 'no antidote prompt is possible')

    def test_later_nights_may_save_herself(self):
        self.state.night.wolf_kills = [0]
        driver = self.run_witch(cycle=2)
        self.assertEqual(driver.legal['save'], [0])

    def test_one_potion_each_per_game(self):
        action = self.state.players[0].job_actions
        self.assertEqual((action.poison_left, action.antidote_left), (1, 1))
        action.execute(self.state, self.state.players[0], kill=7)
        self.assertEqual(action.poison_left, 0)
        action.execute(self.state, self.state.players[0], save=5)
        self.assertEqual(action.antidote_left, 0)
        # a fresh night must not refill them
        action.reset()
        self.assertEqual((action.poison_left, action.antidote_left), (0, 0))

    def test_poison_cannot_be_revived(self):
        state = support.new_state(jobs={0: Jobs.Witch})
        state.night.wolf_kills = [5]
        state.night.witch_saved_target = 5
        state.night.witch_killed_target = 5
        engine.GameRuleDefinitions.end_night(state)
        self.assertIs(state.players[5].player_state, PlayerState.Dead)


class Guard(unittest.TestCase):
    def guard_targets(self, guarded, cycle=2):
        state = support.new_state(cycle=cycle, jobs={0: Jobs.Guard})
        state.guarded_players = list(guarded)
        with support.Driver() as driver:
            engine.GameRuleDefinitions.guard_do_night(state)
        return driver.legal['target']

    def test_may_guard_himself(self):
        self.assertIn(0, self.guard_targets([]))

    def test_cannot_guard_the_same_player_two_nights_running(self):
        self.assertNotIn(4, self.guard_targets([4]))
        self.assertIn(4, self.guard_targets([9]), 'only the previous night is barred')

    def test_cannot_guard_the_out(self):
        state = support.new_state(jobs={0: Jobs.Guard})
        support.exited(state, 6)
        with support.Driver() as driver:
            engine.GameRuleDefinitions.guard_do_night(state)
        self.assertNotIn(6, driver.legal['target'])

    def test_guard_blocks_the_wolf_kill(self):
        state = support.new_state(jobs={0: Jobs.Guard})
        state.night.wolf_kills = [5]
        state.night.guard_target = 5
        engine.GameRuleDefinitions.end_night(state)
        self.assertIs(state.players[5].player_state, PlayerState.Alive)

    def test_guard_does_not_stop_the_poison(self):
        state = support.new_state(jobs={0: Jobs.Guard})
        state.night.guard_target = 5
        state.night.witch_killed_target = 5
        engine.GameRuleDefinitions.end_night(state)
        self.assertIs(state.players[5].player_state, PlayerState.Dead)

    def test_guarded_and_saved_together_dies(self):
        state = support.new_state(jobs={0: Jobs.Guard})
        state.night.wolf_kills = [5]
        state.night.guard_target = 5
        state.night.witch_saved_target = 5
        engine.GameRuleDefinitions.end_night(state)
        self.assertIs(state.players[5].player_state, PlayerState.Dead, '同守同救判定死亡')

    def test_saving_alone_rescues(self):
        state = support.new_state(jobs={0: Jobs.Witch})
        state.night.wolf_kills = [5]
        state.night.witch_saved_target = 5
        engine.GameRuleDefinitions.end_night(state)
        self.assertIs(state.players[5].player_state, PlayerState.Alive)


class Shots(unittest.TestCase):
    def test_hunter_shoots_once_when_killed_by_wolves(self):
        state = support.new_state(jobs={7: Jobs.Hunter})
        support.kill(state, 7)
        state.night.wolf_kills = [7]
        with support.Driver() as driver:
            engine.GameRuleDefinitions.begin_day(state)
        self.assertEqual(driver.shoot_prompts(), 1)

    def test_hunter_cannot_shoot_another_player_who_died_tonight(self):
        '''
        rule.md: the hunter may shoot "任意一个存活玩家".

        On the dawn after a multi-death night the other victims are still
        PlayerState.Dead (they only become Exit as begin_day walks the list),
        so the target pool must be built from Alive, not exited_players.
        '''
        state = support.new_state(jobs={3: Jobs.Hunter})
        support.kill(state, 3)   # the hunter, killed by the wolves
        support.kill(state, 9)   # another victim of the same night
        state.night.wolf_kills = [3, 9]

        with support.Driver(pass_rate=0.0) as driver:
            # Aim at a living seat so the only thing under test is the pool.
            driver.pick = lambda key, legal, voter, prompt: (
                10 if 'shoot' in prompt else legal[0])
            engine.GameRuleDefinitions.begin_day(state)

        self.assertNotIn(9, driver.legal['ID'],
                         'the other player who died tonight is not alive')
        self.assertNotIn(3, driver.legal['ID'], 'the hunter cannot shoot himself')
        self.assertIn(10, driver.legal['ID'], 'living seats stay legal targets')
        self.assertIsNot(state.players[9].out_reason, OutReason.Shot_By_Hunter)

    def test_hunter_does_not_shoot_when_poisoned(self):
        state = support.new_state(jobs={7: Jobs.Hunter})
        support.kill(state, 7)
        state.night.witch_killed_target = 7
        with support.Driver() as driver:
            engine.GameRuleDefinitions.begin_day(state)
        self.assertEqual(driver.shoot_prompts(), 0, '被毒死不能开枪')

    def test_hunter_shoots_once_when_exiled(self):
        state = support.new_state(stage=engine.GameStage.Day, jobs={4: Jobs.Hunter})
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'n'
            driver.pick = vote_to(4)
            engine.GameRuleDefinitions.vote_and_exile_day(state)
        self.assertEqual(driver.shoot_prompts(), 1)

    def test_wolf_king_shoots_only_when_exiled(self):
        state = support.new_state(stage=engine.GameStage.Day, jobs={4: Jobs.Wolf_King})
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'n'
            driver.pick = vote_to(4)
            engine.GameRuleDefinitions.vote_and_exile_day(state)
        self.assertEqual(driver.shoot_prompts(), 1)

        state = support.new_state(jobs={4: Jobs.Wolf_King})
        support.kill(state, 4)
        state.night.wolf_kills = [4]
        with support.Driver() as driver:
            engine.GameRuleDefinitions.begin_day(state)
        self.assertEqual(driver.shoot_prompts(), 0, '狼王被刀死不能开枪')

    def test_a_shot_kills_exactly_one_extra_player(self):
        # no abstentions: the tally has to be exactly what the test expects
        state = support.new_state(stage=engine.GameStage.Day, jobs={4: Jobs.Hunter})
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'n'
            driver.pick = vote_to(4, shoot=9)
            engine.GameRuleDefinitions.vote_and_exile_day(state)
        self.assertEqual(sorted(driver.exits), sorted([(4, 'Voted'), (9, 'Shot_By_Hunter')]))


class WinConditions(unittest.TestCase):
    def outcome(self, state):
        engine.GameRuleDefinitions.determine_game_process_day(state)
        return state.stage

    def test_civilians_win_when_every_wolf_is_out(self):
        state = support.new_state(jobs={2: Jobs.Wolf, 3: Jobs.Wolf_King})
        support.exited(state, 2, 3)
        self.assertIs(self.outcome(state), engine.GameStage.End_Civilian_Win)

    def test_wolves_win_when_all_gods_are_out(self):
        state = support.new_state(jobs={1: Jobs.Prophet, 2: Jobs.Witch,
                                        3: Jobs.Guard, 4: Jobs.Hunter})
        support.exited(state, 1, 2, 3, 4)
        self.assertIs(self.outcome(state), engine.GameStage.End_Wolf_Win)

    def test_wolves_win_when_all_civilians_are_out(self):
        state = support.new_state(jobs={2: Jobs.Wolf})
        support.exited(state, *(i for i, p in state.players.items() if p.job is Jobs.Civilian))
        self.assertIs(self.outcome(state), engine.GameStage.End_Wolf_Win)

    def test_empty_camp_does_not_end_the_game(self):
        # a state with no civilians at all must not satisfy "all civilians dead"
        state = support.new_state(jobs={i: Jobs.Wolf for i in range(engine.PLAYERS_COUNT)})
        self.assertIsNot(self.outcome(state), engine.GameStage.End_Wolf_Win)

    def test_tie_after_the_cycle_limit(self):
        state = support.new_state(cycle=engine.MAXIMUM_CYCLES + 1)
        self.assertIs(self.outcome(state), engine.GameStage.End_Tie)


class SelfExplode(unittest.TestCase):
    def explode(self, state, driver, who):
        def yesno(prompt, player):
            return 'y' if 'expose' in prompt and player == who else 'n'
        driver.yesno = yesno
        engine.GameRuleDefinitions.comment_day(state)
        return driver

    def test_speech_is_kept_and_the_day_is_cut_short(self):
        state = support.new_state(cycle=1, jobs={0: Jobs.Wolf, 1: Jobs.Wolf})
        with support.Driver() as driver:
            self.explode(state, driver, 0)
        self.assertIn(0, state.day.messages, 'his speech must be kept')
        self.assertIn(0, state.exited_players)
        self.assertEqual(state.players[0].out_reason, OutReason.Identification_Exposed)
        later = [p for p in driver.spoke if p not in (0,)]
        self.assertEqual(later, [], '跳过后续玩家的发言')
        self.assertIs(state.stage, engine.GameStage.Night, 'and the vote is skipped')

    def test_the_wolf_king_may_explode_too(self):
        state = support.new_state(cycle=1, jobs={0: Jobs.Wolf_King, 1: Jobs.Wolf})
        with support.Driver() as driver:
            self.explode(state, driver, 0)
        self.assertIn(0, state.exited_players)

    def test_two_explosions_remove_the_badge(self):
        state = support.new_state(cycle=3, jobs={0: Jobs.Wolf, 1: Jobs.Wolf})
        state.wolf_exposed = [1]
        state.day.the_police = 5
        state.day.break_the_police = False
        with support.Driver() as driver:
            self.explode(state, driver, 0)
        self.assertIsNone(state.day.the_police)
        self.assertTrue(state.day.break_the_police)

    def test_one_explosion_keeps_the_badge(self):
        state = support.new_state(cycle=3, jobs={0: Jobs.Wolf, 1: Jobs.Wolf})
        state.day.the_police = 5
        state.day.break_the_police = False
        with support.Driver() as driver:
            self.explode(state, driver, 0)
        self.assertEqual(state.day.the_police, 5)

    def test_an_explosion_that_ends_the_game_stays_ended(self):
        state = support.new_state(cycle=4, jobs={0: Jobs.Wolf})
        for player_id in range(1, engine.PLAYERS_COUNT):
            support.exited(state, player_id)
        with support.Driver() as driver:
            self.explode(state, driver, 0)
        self.assertTrue(state.stage.name.startswith('End_'),
                        f'a finished game must not roll into another night ({state.stage})')

    def test_a_dead_wolf_giving_last_words_is_not_offered_an_explosion(self):
        state = support.new_state(cycle=1, jobs={0: Jobs.Wolf})
        support.kill(state, 0)
        with support.Driver() as driver:
            engine.GameRuleDefinitions.begin_day(state)
            engine.GameRuleDefinitions.deaths_comment_day(state)
        self.assertFalse([p for p in driver.prompts if 'expose' in p[0]],
                         '遗言 is a plain statement')


class Police(unittest.TestCase):
    def elect(self, candidates, ballots):
        # no abstentions: the tally has to be exactly what the test expects
        '''ballots(voter) -> the candidate that voter backs'''
        state = support.new_state(cycle=1, stage=engine.GameStage.Day)
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'y' if player in candidates else 'n'
            driver.pick = lambda key, legal, voter, prompt: ballots(voter)
            engine.GameRuleDefinitions.vote_police(state)
        return state

    def test_one_candidate_becomes_police_without_a_vote(self):
        state = self.elect({2}, lambda voter: 2)
        self.assertEqual(state.day.the_police, 2)

    def test_no_candidate_means_no_police(self):
        state = self.elect(set(), lambda voter: 0)
        self.assertIsNone(state.day.the_police)

    def test_a_tie_means_no_police(self):
        # ten voters split evenly: 5 for P2, 5 for P5
        state = self.elect({2, 5}, lambda voter: 2 if voter % 2 else 5)
        self.assertIsNone(state.day.the_police)
        self.assertEqual(set(state.day.police_voting_result.values()), {5, 5})

    def test_only_non_candidates_vote(self):
        # no abstentions: the tally has to be exactly what the test expects
        state = support.new_state(cycle=1, stage=engine.GameStage.Day)
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'y' if player in (2, 5) else 'n'
            driver.pick = lambda key, legal, voter, prompt: 2
            engine.GameRuleDefinitions.vote_police(state)
        voters = {player for text, player, _, _ in driver.prompts if 'Vote whom' in text}
        self.assertEqual(voters, set(state.day.police_vote_person_camp[False]))
        self.assertFalse(voters & {2, 5}, 'candidates do not vote')


class Sheriff(unittest.TestCase):
    def test_the_sheriff_has_one_and_a_half_votes(self):
        # no abstentions: the tally has to be exactly what the test expects
        state = support.new_state(stage=engine.GameStage.Day, jobs={4: Jobs.Prophet})
        state.day.the_police = 4
        state.day.break_the_police = False
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'n'
            driver.pick = vote_to(4)
            engine.GameRuleDefinitions.vote_and_exile_day(state)
        self.assertEqual(state.day.vote_result[4][0], 11 + 1.5)

    def test_the_badge_passes_only_to_a_living_player(self):
        state = support.new_state(jobs={4: Jobs.Prophet})
        state.day.the_police = 4
        state.day.break_the_police = False
        support.kill(state, 4)
        support.kill(state, 5)
        support.exited(state, 6)
        state.night.wolf_kills = [4, 5]
        with support.Driver() as driver:
            driver.yesno = lambda prompt, player: 'y'
            driver.pick = lambda key, legal, voter, prompt: 8
            engine.GameRuleDefinitions.begin_day(state)
        self.assertEqual(state.day.the_police, 8)
        legal = driver.legal['ID']
        self.assertNotIn(5, legal, 'a player who died tonight cannot take the badge')
        self.assertNotIn(6, legal, 'nor one who is already out')

    def test_the_badge_is_handled_when_the_sheriff_is_exiled(self):
        state = support.new_state(stage=engine.GameStage.Day, jobs={4: Jobs.Prophet})
        state.day.the_police = 4
        state.day.break_the_police = False
        with support.Driver(pass_rate=0.0) as driver:
            driver.yesno = lambda prompt, player: 'y' if 'badge' in prompt else 'n'
            driver.pick = vote_to(4, shoot=None)
            engine.GameRuleDefinitions.vote_and_exile_day(state)
        self.assertIn(4, state.exited_players)
        self.assertNotEqual(state.day.the_police, 4, 'a dead sheriff cannot keep the badge')

    def test_skipping_the_hand_over_tears_the_badge(self):
        state = support.new_state(jobs={4: Jobs.Prophet})
        state.day.the_police = 4
        state.day.break_the_police = False
        support.kill(state, 4)
        state.night.wolf_kills = [4]
        with support.Driver(pass_rate=1.0) as driver:
            driver.yesno = lambda prompt, player: 'y'
            driver.pass_rate = 1.0        # every optional prompt is skipped
            engine.GameRuleDefinitions.begin_day(state)
        self.assertIsNone(state.day.the_police)
        self.assertTrue(state.day.break_the_police)


if __name__ == '__main__':
    unittest.main()
