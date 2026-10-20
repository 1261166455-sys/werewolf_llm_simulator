'''
Whole-game invariants: the properties every game must satisfy, whatever the
players decide.

These are the properties that broke silently in the past, so each one names the
rule it protects.
'''
from __future__ import annotations

import collections
import unittest

import support
import engine

SEEDS = 20


class WholeGames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = []
        with support.Driver() as driver:
            for seed in range(SEEDS):
                driver.play(seed, pass_rate=0.2 if seed % 2 else 0.0)
                cls.games.append((seed, driver.state,
                                  list(driver.prompts), list(driver.exits)))

    def test_every_game_terminates_in_a_real_ending(self):
        endings = collections.Counter(state.stage.name for _, state, _, _ in self.games)
        for stage in endings:
            self.assertIn(stage, ('End_Civilian_Win', 'End_Wolf_Win', 'End_Tie'))
        self.assertEqual(sum(endings.values()), SEEDS)

    def test_role_counts_match_the_rule(self):
        for seed, state, _, _ in self.games:
            counts = collections.Counter(p.job for p in state.players.values())
            self.assertEqual(dict(counts), engine.GameRule.Roles, f'seed {seed}')

    def test_only_three_things_reach_a_non_alive_player(self):
        '''
        A player who is not Alive may still be asked:
          * 遗言, the day-1 last words (they are already Exit by then)
          * their dying shot, which exit_player fires after marking them out
          * whether the badge passes on, asked of the dying sheriff
        Anything else means a corpse is being made to act.
        '''
        for seed, _, prompts, _ in self.games:
            for text, player, cycle, status in prompts:
                if status == 'Alive':
                    continue
                last_words = text == '[Addresser]' and cycle == 1
                dying_action = 'shoot' in text or 'badge' in text
                self.assertTrue(last_words or dying_action,
                                f'seed {seed}: P{player} ({status}) was asked {text!r} on day {cycle}')

    def test_every_exit_is_recorded_once_with_a_reason(self):
        for seed, state, _, exits in self.games:
            players = [player for player, _ in exits]
            self.assertEqual(len(players), len(set(players)), f'seed {seed}: an exit fired twice')
            self.assertEqual(sorted(players), sorted(state.exited_players), f'seed {seed}')
            for player, reason in exits:
                self.assertEqual(state.players[player].out_reason.name, reason, f'seed {seed}')
                self.assertIs(state.players[player].player_state, engine.PlayerState.Exit)

    def test_history_holds_one_entry_per_finished_cycle(self):
        for seed, state, _, _ in self.games:
            self.assertEqual(len(state.history), state.cycles - 1, f'seed {seed}')

    def test_the_police_is_elected_only_on_the_first_day(self):
        with support.Driver() as driver:
            elections = []
            original = engine.GameRuleDefinitions.vote_police

            def watched(state):
                elections.append(state.cycles)
                return original(state)

            engine.GameRuleDefinitions.vote_police = staticmethod(watched)
            try:
                for seed in range(6):
                    driver.play(seed)
            finally:
                engine.GameRuleDefinitions.vote_police = original
        self.assertEqual(len(elections), 6, 'one election per game')
        self.assertEqual(set(elections), {1}, '警长的竞选仅发生在游戏第一天的白天')

    def test_the_wolf_victim_list_has_no_duplicates(self):
        for seed, state, _, _ in self.games:
            for night, _ in state.history:
                kills = night.wolf_kills or []
                self.assertEqual(len(kills), len(set(kills)), f'seed {seed}: {kills}')

    def test_the_pack_kills_at_most_one_player_a_night(self):
        for seed, state, _, _ in self.games:
            for night, _ in state.history:
                self.assertLessEqual(len(night.wolf_kills or []), 1, f'seed {seed}')


class SeatBounds(unittest.TestCase):
    def probe(self, *answers):
        with support.Driver() as driver:
            driver.queue[:] = list(answers)
            return driver._orig['an_id']('probe ', True, 0, 'ID', ())

    def test_last_seat_is_legal(self):
        self.assertEqual(self.probe('11'), {'ID': 11})

    def test_first_seat_is_legal(self):
        self.assertEqual(self.probe('0'), {'ID': 0})

    def test_out_of_range_is_rejected_and_re_asked(self):
        self.assertEqual(self.probe('12', '5'), {'ID': 5})
        self.assertEqual(self.probe('-1', '7'), {'ID': 7})

    def test_excluded_seat_is_rejected_and_re_asked(self):
        with support.Driver() as driver:
            driver.queue[:] = ['3', '4']
            self.assertEqual(driver._orig['an_id']('probe ', True, 0, 'ID', (3,)), {'ID': 4})


class PhaseOrder(unittest.TestCase):
    def test_night_phases_run_in_the_configured_order(self):
        with support.Driver() as driver:
            calls = []
            originals = {name: getattr(engine.GameRuleDefinitions, name)
                         for name in (f'{p}_night' for p in engine.GameRule.Night)}
            for name, fn in originals.items():
                def watched(state, _fn=fn, _name=name):
                    calls.append(_name)
                    return _fn(state)
                setattr(engine.GameRuleDefinitions, name, staticmethod(watched))
            try:
                state = driver.play(3)
            finally:
                for name, fn in originals.items():
                    setattr(engine.GameRuleDefinitions, name, staticmethod(fn))
        expected = [f'{p}_night' for p in engine.GameRule.Night] * state.cycles
        self.assertEqual(calls, expected)

    def test_speeches_come_before_the_vote_each_day(self):
        with support.Driver() as driver:
            order: list[tuple[int, str]] = []
            speak = engine.GameRuleDefinitions.comment_day
            vote = engine.GameRuleDefinitions.vote_and_exile_day

            def watched_speak(state):
                order.append((state.cycles, 'speak'))
                return speak(state)

            def watched_vote(state):
                order.append((state.cycles, 'vote'))
                return vote(state)

            engine.GameRuleDefinitions.comment_day = staticmethod(watched_speak)
            engine.GameRuleDefinitions.vote_and_exile_day = staticmethod(watched_vote)
            try:
                driver.play(5)
            finally:
                engine.GameRuleDefinitions.comment_day = staticmethod(speak)
                engine.GameRuleDefinitions.vote_and_exile_day = staticmethod(vote)

        by_cycle: dict[int, list[str]] = {}
        for cycle, phase in order:
            by_cycle.setdefault(cycle, []).append(phase)
        for cycle, phases in by_cycle.items():
            if 'vote' in phases:
                self.assertIn('speak', phases, f'day {cycle} voted without speaking')
                self.assertLess(phases.index('speak'), phases.index('vote'),
                                f'day {cycle}: speech must come before the vote')


if __name__ == '__main__':
    unittest.main()
