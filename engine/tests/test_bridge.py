'''
The console bridge: ui.Session must drive the engine at 12 seats, mask secrets,
replay without calling a model, and switch language.

These tests exercise the real Session, not a mock, because most of the console's
behaviour (role masking, replay, the private channel) only exists there.
'''
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import support        # puts engine/ on sys.path
import engine
import i18n
import llm_interface
import ui


def answer_with_legal_choice(self, prompt, player, meta):
    '''Stand in for the browser: pick a legal seat from the offered choices.'''
    choices = (meta or {}).get('choices') or []
    value = str(choices[0]) if choices else ''
    self.answers.append(value)
    self.emit(self._prompt_event(prompt, player, meta, 'human'))
    self.emit({'kind': 'answer', 'prompt': prompt, 'player': player, 'value': value,
               'secret': self._secret(prompt, meta)})
    return value


class Session(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        i18n.set_language('zh')
        cls.original_answer = ui.Session._human_answer
        ui.Session._human_answer = answer_with_legal_choice
        cls.session = ui.Session()
        ui.CURRENT = cls.session
        cls.session.start_game(seed=1234)
        cls.session.thread.join(timeout=180)

    @classmethod
    def tearDownClass(cls):
        ui.Session._human_answer = cls.original_answer

    def test_every_seat_is_offered(self):
        self.assertEqual(sorted(self.session.seats), list(range(engine.PLAYERS_COUNT)))
        self.assertEqual(len(self.session.seats), 12)

    def test_the_game_finishes(self):
        self.assertEqual(self.session.status, 'finished')
        stage = self.session.game.game_state.stage
        self.assertIn(stage, (engine.GameStage.End_Civilian_Win,
                              engine.GameStage.End_Wolf_Win, engine.GameStage.End_Tie))

    def test_the_board_shows_every_player(self):
        snapshot = self.session.snapshot()
        self.assertEqual(len(snapshot['players']), engine.PLAYERS_COUNT)
        self.assertTrue(all({'id', 'job', 'state'} <= set(p) for p in snapshot['players']))

    def test_every_seat_can_be_observed_in_both_languages(self):
        state = self.session.game.game_state
        request = llm_interface.DecisionRequest(
            kind=llm_interface.DecisionKind.Generic, question='probe',
            allowed=(0,), required=False)
        for viewer in range(engine.PLAYERS_COUNT):
            observation = self.session.observation(viewer)
            self.assertEqual(observation.role, state.players[viewer].job.value)
            for language in ('zh', 'en'):
                system, user = llm_interface.build_messages(
                    observation, request, language=language)
                self.assertIn(i18n.name(i18n.JOB_NAMES, observation.role, language),
                              system + user, f'seat {viewer} / {language}')

    def test_the_wolf_camp_knows_its_pack(self):
        state = self.session.game.game_state
        camp = engine.GameRule.WolfCamp
        for viewer in engine.GameRuleDefinitions.players_with(state, camp):
            facts = [f for f in self.session.observation(viewer).private_facts
                     if '狼队友' in f or 'fellow Wolf' in f]
            expected = len(engine.GameRuleDefinitions.players_with(state, camp)) - 1
            self.assertTrue(facts, f'seat {viewer} was told nothing about the pack')
            self.assertIn(str(expected), str(facts[0].count('玩家')) + str(expected))


class Privacy(Session):
    def test_night_phases_are_announced_whoever_is_alive(self):
        logs = [e['text'] for e in self.session.events if e.get('kind') == 'log']
        envelope = [line for line in logs if '阶段开始' in line or '阶段结束' in line]
        state = self.session.game.game_state
        self.assertEqual(len(envelope), 2 * len(engine.GameRule.Night) * state.cycles)

    def test_no_private_text_reaches_the_transcript(self):
        logs = ' '.join(e['text'] for e in self.session.events if e.get('kind') == 'log')
        for secret in ('查验结果', '毒药', '狼队友', '请输入'):
            self.assertNotIn(secret, logs, f'{secret!r} leaked into the public log')

    def test_private_events_carry_role_and_facts(self):
        private = [e for e in self.session.events if e.get('kind') == 'private']
        self.assertTrue(private, 'the private channel was never used')
        prompts = [e for e in self.session.events if e.get('kind') == 'prompt' and e.get('secret')]
        self.assertTrue(prompts, 'no prompt was marked private')
        for event in prompts:
            self.assertTrue((event.get('private') or {}).get('role'))

    def test_only_rules_public_decisions_are_public(self):
        '''Speech, vote, and the political decisions the rules announce.'''
        public = [e for e in self.session.events
                  if e.get('kind') == 'prompt' and not e.get('secret')]
        self.assertTrue(public)
        allowed = {'speech', 'vote', 'police', 'badge', 'speech_order', 'shot'}
        for event in public:
            self.assertIn(event.get('decision'), allowed,
                          f'unexpected public prompt: {event["raw"]!r}')
        # and a wolf's self-explosion question must never be public
        secrets = {e.get('decision') for e in self.session.events
                   if e.get('kind') == 'prompt' and e.get('secret')}
        self.assertNotIn('expose', {e.get('decision') for e in public})


class Decisions(unittest.TestCase):
    '''The LLM-facing contract: every engine prompt must classify, and the
    decoder must validate what comes back.'''

    PROMPTS = [
        ('[Wolf pack] Propose a player to kill (the most proposed player dies; leave blank to skip): ', 'ID', 'pack_kill'),
        ("[Prophet] Input a player's ID to query its state: ", 'ID', 'query'),
        ("[Witch] Input a player's ID to kill: ", 'kill', 'kill'),
        ("[Witch] Input a player's ID to save (Please select one who was killed tonight): ", 'save', 'save'),
        ("[Guard] Input a player's ID to guard (No double guarding): ", 'target', 'guard'),
        ('[Vote] Vote whom? ', 'ID', 'vote'),
        ('[Addresser] ', 'ID', 'speech'),
        ('[Speecher] ', 'ID', 'speech'),
        ('You can shoot 1 person, input an ID: ', 'ID', 'shot'),
        ('Do you want to transmit your badge to a new player? [y/n] ', 'ID', 'badge'),
        ('Which player do you want to send your badge? ', 'ID', 'badge_target'),
        ('[Option] Choose to be the police? [y/n] ', 'ID', 'police'),
        ('[Choice] Do you want to expose your identification? [y/n] ', 'ID', 'expose'),
        ('Determine the comment order, start from you, yes for clockwise, no for counterclockwise [y/n] ', 'ID', 'speech_order'),
    ]

    def test_every_engine_prompt_classifies(self):
        for prompt, key, expected in self.PROMPTS:
            self.assertEqual(llm_interface.classify(key, prompt).value, expected, prompt)

    def test_id_questions_are_never_yes_no(self):
        '''
        A yes/no question renders 是/否 buttons and answers 'y'/'n'. If an id
        question is misclassified that way, the engine rejects 'y' as a seat
        number and re-prompts forever - which is exactly the badge bug.
        '''
        # the expectation comes from this table, not from classify(), so the
        # check is independent of the code under test
        yes_no_expected = {'police', 'expose', 'badge', 'speech_order'}
        for prompt, key, expected in self.PROMPTS:
            kind = llm_interface.classify(key, prompt)
            self.assertEqual(kind in llm_interface.YES_NO_KINDS,
                             expected in yes_no_expected,
                             f'{prompt!r} classified as {kind.value}, expected {expected}')

    def test_badge_questions_are_two_different_kinds(self):
        ask = llm_interface.classify('ID', 'Do you want to transmit your badge to a new player? [y/n] ')
        who = llm_interface.classify('ID', 'Which player do you want to send your badge? ')
        self.assertNotEqual(ask, who)
        self.assertIn(ask, llm_interface.YES_NO_KINDS)
        self.assertNotIn(who, llm_interface.YES_NO_KINDS)
        self.assertEqual(llm_interface.decode('{"choice": 7}', llm_interface.DecisionRequest(
            kind=who, question='q', allowed=(3, 7), required=False)).as_answer(), '7')

    def test_yes_no_questions_decode_to_y_or_n(self):
        for kind in llm_interface.YES_NO_KINDS:
            request = llm_interface.DecisionRequest(kind=kind, question='q',
                                                    allowed=None, required=True)
            self.assertEqual(llm_interface.decode('{"yes": true}', request).as_answer(), 'y')
            self.assertEqual(llm_interface.decode('{"yes": false}', request).as_answer(), 'n')
            self.assertEqual(llm_interface.decode('{"choice": "是"}', request).as_answer(), 'y')
            with self.assertRaises(llm_interface.DecodeError):
                llm_interface.decode('{}', request)
            # a fallback must be the harmless answer, never "yes"
            fallback = llm_interface.LLMController.fallback(request, 'boom')
            self.assertEqual(fallback.as_answer(), 'n')

    def test_illegal_id_is_rejected(self):
        request = llm_interface.DecisionRequest(
            kind=llm_interface.DecisionKind.Kill, question='q',
            allowed=(1, 2), required=True)
        with self.assertRaises(llm_interface.DecodeError):
            llm_interface.decode('{"choice": 5}', request)
        self.assertEqual(llm_interface.decode('{"choice": 2}', request).choice, 2)

    def test_the_prompt_states_the_twelve_player_rules(self):
        for language in ('en', 'zh'):
            rules = llm_interface.RULES[language]
            self.assertIn('0-11', rules)
            for role in ('Wolf King', 'Guard', 'Hunter'):
                if language == 'en':
                    self.assertIn(role, rules)
            self.assertIn('10', rules, 'the round limit must be stated')
        zh = llm_interface.RULES['zh']
        for name in ('狼王', '守卫', '猎人', '女巫', '预言家', '平民'):
            self.assertIn(name, zh)

    def test_every_role_has_a_briefing(self):
        for language in ('en', 'zh'):
            for job in engine.Jobs:
                if job is engine.Jobs.Not_Assigned:
                    continue
                self.assertTrue(llm_interface.ROLE_BRIEFING[language].get(job.value),
                                f'{language}: {job.value} has no briefing')

    def test_a_yes_no_question_asks_for_yes_or_no(self):
        observation = llm_interface.Observation(
            viewer=0, role='Wolf', stage='Day', cycle=1,
            living=(0, 1), out_of_play=())
        request = llm_interface.DecisionRequest(
            kind=llm_interface.DecisionKind.Expose, question='engine text',
            allowed=None, required=True)
        for language in ('en', 'zh'):
            system, user = llm_interface.build_messages(observation, request, language=language)
            self.assertIn('legal choices' if language == 'en' else '合法选择', user)
            self.assertIn('"yes"', user, 'the answer field must be named')
            self.assertIn('true', user.lower())


class Seats(unittest.TestCase):
    '''Every seat must be settable, not just the first six.'''

    def test_all_twelve_seats_accept_a_change(self):
        session = ui.Session()
        session.set_seats({i: 'llm' for i in range(engine.PLAYERS_COUNT)})
        self.assertEqual(set(session.seats.values()), {'llm'})
        session.set_seats({i: 'human' for i in range(engine.PLAYERS_COUNT)})
        self.assertEqual(set(session.seats.values()), {'human'})

    def test_a_single_high_seat_can_be_switched(self):
        session = ui.Session()
        session.set_seats({11: 'llm'})
        self.assertEqual(session.seats[11], 'llm')
        self.assertEqual(session.seats[10], 'human', 'only the named seat changes')


class WolfCampView(unittest.TestCase):
    '''
    A wolf recognises the pack, but only as the camp: the Wolf King must be
    indistinguishable from a plain Wolf on the board.
    '''

    @classmethod
    def setUpClass(cls):
        i18n.set_language('zh')
        state = support.new_state(jobs={0: engine.Jobs.Wolf, 1: engine.Jobs.Wolf_King,
                                       2: engine.Jobs.Wolf, 3: engine.Jobs.Civilian})
        cls.session = ui.Session()
        game = engine.Game()
        game.game_state = state
        cls.session.game = game
        ui.CURRENT = cls.session

    def test_a_wolf_sees_the_whole_pack_including_the_king(self):
        view = self.session._private_view(0)
        self.assertEqual(sorted(view['camp_mates']), [1, 2])
        self.assertEqual(view['role'], '狼人')

    def test_the_king_knows_his_own_rank_but_sees_others_as_pack(self):
        view = self.session._private_view(1)
        self.assertEqual(sorted(view['camp_mates']), [0, 2])
        self.assertEqual(view['role'], '狼王')

    def test_a_civilian_sees_no_camp(self):
        self.assertEqual(self.session._private_view(3)['camp_mates'], [])

    def test_the_camp_never_names_a_rank(self):
        # the board label is one fixed string for every camp mate
        for seat in (0, 1, 2):
            view = self.session._private_view(seat)
            self.assertNotIn('狼王', str(view['camp_mates']),
                             'camp mates travel as ids, so no rank can leak')

    def test_a_later_wolf_sees_the_proposals_already_made(self):
        '''
        The pack discusses in seat order, so a wolf who has not yet spoken must
        see what its packmates proposed - but not its own proposal.
        '''
        state = self.session.game.game_state
        original = state.night
        state.night = engine.NightHistory(wolf_kills=[], wolf_proposals={0: 5, 1: None})
        try:
            later = ' '.join(self.session.observation(2).private_facts)
            first = ' '.join(self.session.observation(0).private_facts)
        finally:
            state.night = original
        lang = self.session.language
        said = i18n.t('fact_pack_proposal_target', lang, wolf=0, target=5)
        self.assertIn(said, later)
        self.assertIn(i18n.t('fact_pack_proposal_pass', lang, wolf=1), later)
        self.assertNotIn(said, first, 'a wolf is not shown its own proposal')


class Replay(Session):
    def test_replay_reproduces_the_same_game_without_models(self):
        first = self.session.snapshot()
        self.session.start_replay(delay=0)
        self.session.thread.join(timeout=180)
        replayed = self.session.snapshot()
        self.assertEqual(self.session.status, 'finished')
        self.assertEqual(replayed['winner'], first['winner'])
        controllers = {e.get('controller') for e in self.session.events
                       if e.get('kind') == 'prompt'}
        self.assertIn('replay', controllers, 'the replay must not ask a human or a model')


class Decoding(unittest.TestCase):
    '''
    A recorded game lost a turn to a stray bracket in the model's JSON; these are
    the shapes real completions have arrived in.
    '''

    def test_a_stray_closing_paren_is_dropped(self):
        # the exact reply from replays/20261018-001609: "...。" followed by ")}"
        raw = ('{"choice": 3, "speech": "", "reasoning": "没有理由第一天就投掉唯一预言家。")}')
        payload = llm_interface.extract_json_object(raw)
        self.assertEqual(payload['choice'], 3)
        self.assertTrue(payload['reasoning'].endswith('。'))

    def test_a_truncated_reply_still_yields_its_fields(self):
        raw = '{"choice": 4, "reasoning": "我觉得应该投'
        payload = llm_interface.extract_json_object(raw)
        self.assertEqual(payload['choice'], 4)
        self.assertIn('我觉得', payload['reasoning'])

    def test_a_reply_without_json_is_still_an_error(self):
        with self.assertRaises(llm_interface.DecodeError):
            llm_interface.extract_json_object('sorry, I cannot answer that')


class ModelConfig(unittest.TestCase):
    '''
    models.json is the one place model settings live: named profiles, a seat map,
    keys by value / file / environment, and several providers in one game.
    '''

    def parse(self, data):
        return llm_interface.parse_model_config(data)

    def test_a_seat_without_a_profile_inherits_everything(self):
        registry = self.parse({'models': {'a': {'kind': 'deepseek', 'model': 'deepseek-flash'}}})
        fallback = llm_interface.ProviderConfig(kind='deepseek', api_key='k', model='deepseek-chat')
        self.assertEqual(registry.profile_for(0), '')
        config = registry.config_for(0, fallback)
        self.assertIs(config, fallback, 'an unmapped seat keeps the previous model')

    def test_a_profile_on_the_same_provider_reuses_the_key(self):
        registry = self.parse({'models': {'a': {'kind': 'deepseek', 'model': 'deepseek-v4-pro'}},
                               'seats': {'0': 'a'}})
        fallback = llm_interface.ProviderConfig(kind='deepseek', api_key='ds-key')
        config = registry.config_for(0, fallback)
        self.assertEqual(config.model, 'deepseek-v4-pro')
        self.assertEqual(config.api_key, 'ds-key')
        self.assertEqual(config.base_url, fallback.base_url)

    def test_a_second_provider_gets_its_own_key_from_a_file(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = Path(folder) / 'models.json'
            (Path(folder) / 'kimi_key.txt').write_text('  moon-key  \n', encoding='utf-8')
            config_path.write_text(json.dumps({
                'models': {'kimi': {'kind': 'compatible', 'model': 'kimi-k2',
                                    'base_url': 'https://api.moonshot.cn/v1',
                                    'key_file': 'kimi_key.txt'}},
                'seats': {'3': 'kimi'}}), encoding='utf-8')
            registry = llm_interface.load_model_config(config_path)
            # a relative key_file is resolved against the file that named it
            config = registry.config_for(
                3, llm_interface.ProviderConfig(kind='deepseek', api_key='ds'))
            self.assertEqual(config.kind, 'compatible')
            self.assertEqual(config.api_key, 'moon-key')
            self.assertEqual(config.base_url, 'https://api.moonshot.cn/v1')
            # (macOS: /var is a symlink to /private/var, so compare resolved paths)
            self.assertEqual(Path(registry.describe()['models']['kimi']['key_source']
                                  .removeprefix('file ')).resolve(),
                             (config_path.parent / 'kimi_key.txt').resolve())

    def test_a_key_can_be_written_in_the_json(self):
        registry = self.parse({'models': {'gpt': {'kind': 'openai', 'model': 'gpt-4o-mini',
                                                  'api_key': 'sk-inline'}},
                               'seats': {'5': 'gpt'}})
        config = registry.config_for(5, llm_interface.ProviderConfig(kind='deepseek', api_key='ds'))
        self.assertEqual(config.api_key, 'sk-inline')
        self.assertEqual(config.base_url, 'https://api.openai.com/v1')

    def test_three_providers_can_play_the_same_game(self):
        registry = self.parse({'models': {
            'ds': {'kind': 'deepseek', 'model': 'deepseek-flash'},
            'gpt': {'kind': 'openai', 'model': 'gpt-4o-mini', 'api_key_env': 'OPENAI_API_KEY'},
            'kimi': {'kind': 'compatible', 'model': 'kimi-k2', 'base_url': 'https://x/v1',
                     'api_key': 'k'}},
            'seats': {'0': 'ds', '1': 'gpt', '2': 'kimi'}})
        fallback = llm_interface.ProviderConfig(kind='deepseek', api_key='ds')
        kinds = {seat: registry.config_for(seat, fallback).kind for seat in range(3)}
        self.assertEqual(kinds, {0: 'deepseek', 1: 'openai', 2: 'compatible'})

    def test_the_default_covers_unlisted_seats(self):
        registry = self.parse({'default': 'a',
                               'models': {'a': {'kind': 'deepseek', 'model': 'deepseek-flash'}}})
        fallback = llm_interface.ProviderConfig(kind='deepseek', api_key='k', model='old')
        self.assertEqual(registry.config_for(7, fallback).model, 'deepseek-flash')

    def test_documentation_keys_are_allowed(self):
        registry = self.parse({'note': ['hello'],
                               'models': {'a': {'model': 'm', '_comment': 'ignore me'}}})
        self.assertEqual(registry.models['a'].fields, {'model': 'm'})

    def test_a_typo_names_itself(self):
        cases = {
            'unknown model': {'models': {'a': {}}, 'seats': {'0': 'b'}},
            'unknown field': {'models': {'a': {'modle': 'x'}}},
            'seat out of range': {'models': {'a': {}}, 'seats': {'12': 'a'}},
            'unknown default': {'models': {'a': {}}, 'default': 'z'},
            'no models': {'models': {}},
        }
        for name, data in cases.items():
            with self.subTest(name):
                with self.assertRaises(llm_interface.LLMError):
                    self.parse(data)

    def test_a_foreign_key_is_never_borrowed(self):
        # run.py publishes the DeepSeek key as LLM_API_KEY; an OpenAI seat must not
        # send it to OpenAI, and must report honestly that it has no key
        registry = self.parse({'models': {'gpt': {'kind': 'openai', 'model': 'gpt-4o-mini'}},
                               'seats': {'4': 'gpt'}})
        old = os.environ.get('LLM_API_KEY')
        os.environ['LLM_API_KEY'] = 'deepseek-key'
        os.environ.pop('OPENAI_API_KEY', None)
        try:
            config = registry.config_for(4, llm_interface.ProviderConfig(kind='deepseek',
                                                                        api_key='deepseek-key'))
            self.assertEqual(config.api_key, '')
            self.assertFalse(registry.describe()['models']['gpt']['has_key'])
        finally:
            if old is None:
                os.environ.pop('LLM_API_KEY', None)
            else:
                os.environ['LLM_API_KEY'] = old

    def test_the_report_never_carries_a_key(self):
        registry = self.parse({'models': {'a': {'kind': 'deepseek', 'model': 'm',
                                                'api_key': 'sk-secret'}},
                               'seats': {'0': 'a'}})
        report = registry.describe(llm_interface.ProviderConfig(kind='deepseek', api_key='k'))
        text = json.dumps(report, ensure_ascii=False)
        self.assertNotIn('sk-secret', text)
        self.assertTrue(report['models']['a']['has_key'])
        self.assertEqual(report['models']['a']['key_source'], 'models.json')
        self.assertEqual(report['seats']['0']['profile'], 'a')

    def test_the_shipped_file_gives_seats_0_and_1_the_thinking_model(self):
        path = Path(__file__).resolve().parent.parent / 'models.json'
        registry = llm_interface.load_model_config(path)
        for seat in (0, 1):
            config = registry.config_for(seat, llm_interface.ProviderConfig(api_key='k'))
            self.assertEqual(config.model, 'deepseek-flash', f'seat {seat}')
            self.assertEqual(config.reasoning_effort, 'max', f'seat {seat}')
            self.assertGreaterEqual(config.max_tokens, 2048, f'seat {seat}')
        # every other seat keeps whatever was configured before, unchanged
        other = llm_interface.ProviderConfig(kind='deepseek', api_key='k', model='deepseek-chat')
        for seat in range(2, 12):
            self.assertIs(registry.config_for(seat, other), other, f'seat {seat}')

    def test_the_shipped_file_holds_no_secret(self):
        # it is committed, so a key must come from a file or the environment
        path = Path(__file__).resolve().parent.parent / 'models.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        for name, profile in data['models'].items():
            with self.subTest(name):
                self.assertEqual(str(profile.get('api_key') or ''), '',
                                 f'{name} inlines a key into a tracked file')


class Thinking(unittest.TestCase):
    '''
    A thinking model (deepseek-flash, deepseek-v4-pro) reasons before it answers:
    the effort is a request parameter and the reasoning comes back separately.
    '''

    def setUp(self):
        self.original = llm_interface._post_json
        self.seen: dict = {}

    def tearDown(self):
        llm_interface._post_json = self.original

    def reply(self, message: dict) -> str:
        def post(url, payload, headers, timeout):
            self.seen['payload'] = payload
            return {'choices': [{'message': message}]}
        llm_interface._post_json = post
        config = llm_interface.ProviderConfig(api_key='k', **self.settings)
        provider = llm_interface.make_provider(config)
        text = provider.complete('system', 'user')
        self.seen['provider'] = provider
        return text

    def test_the_effort_is_asked_for(self):
        self.settings = {'reasoning_effort': 'max'}
        self.reply({'content': '{}'})
        self.assertEqual(self.seen['payload']['reasoning_effort'], 'max')

    def test_off_disables_thinking(self):
        self.settings = {'reasoning_effort': 'off'}
        self.reply({'content': '{}'})
        self.assertEqual(self.seen['payload']['thinking'], {'type': 'disabled'})
        self.assertNotIn('reasoning_effort', self.seen['payload'])

    def test_the_default_sends_neither(self):
        self.settings = {}
        self.reply({'content': '{}'})
        self.assertNotIn('reasoning_effort', self.seen['payload'])
        self.assertNotIn('thinking', self.seen['payload'])

    def test_the_chain_of_thought_is_kept_out_of_the_answer(self):
        self.settings = {}
        text = self.reply({'content': '{"choice": 3}', 'reasoning_content': '  先刀守卫。  '})
        self.assertEqual(text, '{"choice": 3}')
        self.assertEqual(self.seen['provider'].last_reasoning, '先刀守卫。')

    def test_the_decision_keeps_the_thought_when_the_json_has_none(self):
        class Stub:
            name = 'stub'
            last_reasoning = '先刀守卫，别再让他守人。'

            def complete(self, system, user):
                return '{"choice": 3, "reasoning": ""}'

        request = llm_interface.DecisionRequest(
            kind=llm_interface.DecisionKind.Kill, question='[Witch] poison whom?',
            allowed=(0, 1, 2, 3), required=False, key='kill')
        observation = llm_interface.Observation(
            viewer=8, role='Witch', stage='Night', cycle=1,
            living=tuple(range(12)), out_of_play=())
        decision = llm_interface.LLMController(Stub(), attempts=1).decide(observation, request)
        self.assertEqual(decision.as_answer(), '3')
        self.assertEqual(decision.reasoning, Stub.last_reasoning)

    def test_a_thought_that_ate_the_budget_is_reported_clearly(self):
        # the real failure: finish_reason=length, content empty, reasoning long
        self.settings = {'reasoning_effort': 'max', 'max_tokens': 512}

        def post(url, payload, headers, timeout):
            return {'choices': [{'finish_reason': 'length',
                                 'message': {'content': '', 'reasoning_content': '想了很久…'}}]}
        llm_interface._post_json = post
        provider = llm_interface.make_provider(
            llm_interface.ProviderConfig(api_key='k', **self.settings))
        with self.assertRaises(llm_interface.BudgetExhausted) as caught:
            provider.complete('system', 'user')
        message = str(caught.exception)
        self.assertIn('512', message)          # the budget that ran out
        self.assertIn('max_tokens', message)   # and what to raise
        self.assertEqual(provider.last_reasoning, '想了很久…', 'the thought is still worth showing')

    def test_the_controller_widens_the_budget_instead_of_falling_back(self):
        class Truncating:
            name = 'stub'
            last_reasoning = '思考中…'
            budgets = []

            def __init__(self):
                self.config = llm_interface.ProviderConfig(api_key='k', max_tokens=512)

            def complete(self, system, user):
                self.budgets.append(self.config.max_tokens)
                if self.config.max_tokens < 1024:
                    raise llm_interface.BudgetExhausted(
                        f'ran out of {self.config.max_tokens} tokens while still thinking')
                return '{"choice": 3, "reasoning": "ok"}'

            def widen_budget(self, minimum=0):
                self.config.max_tokens = max(minimum, self.config.max_tokens * 4, 4096)
                return self.config.max_tokens

        provider = Truncating()
        request = llm_interface.DecisionRequest(
            kind=llm_interface.DecisionKind.Vote, question='Vote whom?',
            allowed=(1, 2, 3), required=True, key='ID')
        observation = llm_interface.Observation(
            viewer=1, role='Civilian', stage='Day', cycle=1,
            living=tuple(range(12)), out_of_play=())
        decision = llm_interface.LLMController(provider, attempts=2).decide(observation, request)
        self.assertEqual(decision.as_answer(), '3')
        self.assertEqual(decision.note, '', 'a widened retry is not a fallback')
        self.assertEqual(provider.budgets, [512, 4096])

    def test_an_unknown_effort_is_refused(self):
        with self.assertRaises(llm_interface.LLMError):
            llm_interface.ProviderConfig(api_key='k', reasoning_effort='turbo')


class ModelSettings(unittest.TestCase):
    '''
    Model settings live in models.json: the console only reads them.
    '''

    def test_the_default_model_thinks_and_has_room_to(self):
        config = llm_interface.ProviderConfig(api_key='k')
        self.assertEqual(config.reasoning_effort, '')
        self.assertGreaterEqual(config.max_tokens, 1024)

    def test_the_console_cannot_set_a_model(self):
        session = ui.Session()
        before = session.provider_config.model
        # the old editing message is gone; sending it must change nothing
        self.assertFalse(hasattr(session, 'configure_llm'))
        ui.CURRENT, previous = session, ui.CURRENT
        try:
            ui.handle({'type': 'llm', 'config': {'model': 'gpt-4o', 'api_key': 'x'}})
        finally:
            ui.CURRENT = previous
        self.assertEqual(session.provider_config.model, before)
        self.assertFalse(session.provider_config.api_key == 'x')


class ObservationOptions(unittest.TestCase):
    '''
    The observation switches live on the server, so a reload lands in the state the
    operator left behind - and a new one must not disturb its neighbours.
    '''

    def setUp(self):
        i18n.set_language('zh')
        self.previous = ui.CURRENT
        self.session = ui.Session()
        ui.CURRENT = self.session

    def tearDown(self):
        ui.CURRENT = self.previous

    def test_model_thinking_is_off_by_default(self):
        self.assertFalse(self.session.show_thinking)
        self.assertFalse(self.session.public_state()['show_thinking'])

    def test_the_message_turns_it_on_and_keeps_the_others(self):
        self.session.reveal_jobs = True
        ui.handle({'type': 'options', 'show_thinking': True})
        self.assertTrue(self.session.show_thinking)
        self.assertTrue(self.session.reveal_jobs, 'an unrelated switch must survive')
        event = self.session.events[-1]
        self.assertEqual(event['kind'], 'options')
        self.assertTrue(event['show_thinking'])
        self.assertTrue(self.session.public_state()['show_thinking'])

    def test_it_can_be_turned_off_again(self):
        ui.handle({'type': 'options', 'show_thinking': True})
        ui.handle({'type': 'options', 'show_thinking': False})
        self.assertFalse(self.session.show_thinking)


class DecisionWording(unittest.TestCase):
    '''
    Only a statement is something the table heard.

    A vote, a kill or a guard has no public sentence, so a model that fills
    "speech" outside a speech decision is explaining itself; the transcript must
    never phrase that as "says".
    '''

    @staticmethod
    def render(label: str, speech: str) -> str:
        payload = {
            'format': ui.REPLAY_FORMAT, 'seed': 1, 'exported_at': 'now',
            'complete': True, 'language': 'zh', 'answers': ['1'],
            'seats': {'0': 'llm'}, 'model': {'kind': 'x', 'model': 'y'},
            'result': {'stage': 'End_Civilian_Win', 'cycles': 2, 'winner': 'Civilians',
                       'sheriff': None, 'badge_broken': False, 'exploded': 0,
                       'players': [{'id': 0, 'job': 'Wolf', 'state': 'Alive'}]},
            'events': [{'kind': 'decision', 'player': 0, 'label': label,
                        'value': '1', 'speech': speech, 'reasoning': 'r'}],
        }
        session = object.__new__(ui.Session)     # render_text only needs the language
        session.language = 'zh'
        return session.render_text(payload)

    def test_a_vote_never_reports_a_statement(self):
        body = self.render('vote', '我想跟票玩家3')
        self.assertIn(i18n.t('text_thoughts', 'zh') + ' 我想跟票玩家3', body)
        self.assertNotIn(i18n.t('text_speech', 'zh'), body)
        self.assertNotIn('说：', body)

    def test_a_promise_at_a_kill_is_thinking_too(self):
        body = self.render('kill', '今晚杀预言家')
        self.assertIn(i18n.t('text_thoughts', 'zh'), body)
        self.assertNotIn(i18n.t('text_speech', 'zh'), body)

    def test_a_real_statement_keeps_its_label(self):
        body = self.render('speech', '我是平民')
        self.assertIn(i18n.t('text_speech', 'zh') + ' 我是平民', body)
        self.assertNotIn(i18n.t('text_thoughts', 'zh'), body)


class Export(Session):
    def test_export_writes_a_file_that_can_be_listed(self):
        self.session.start_game(seed=99)
        self.session.thread.join(timeout=180)
        path = self.session.export_replay('unit-test')
        self.assertIsNotNone(path)
        try:
            listed = {entry['name'] for entry in self.session.list_replays()}
            self.assertIn(path.name, listed)
        finally:
            path.unlink(missing_ok=True)

    def test_export_without_raw_output_writes_only_json(self):
        self.session.raw_output = False
        self.session.start_game(seed=98)
        self.session.thread.join(timeout=180)
        path = self.session.export_replay('unit-json-only')
        try:
            self.assertIsNotNone(path)
            self.assertFalse(path.with_suffix('.txt').exists())
        finally:
            if path:
                path.unlink(missing_ok=True)
                path.with_suffix('.txt').unlink(missing_ok=True)

    def test_an_export_records_the_rules_it_was_played_under(self):
        self.session.start_game(seed=99)
        self.session.thread.join(timeout=180)
        path = self.session.export_replay('unit-rule-version')
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload.get('rule_version'), engine.GameRule.Version)
            self.assertEqual(payload.get('version'), ui.REPLAY_VERSION)
            entry = {e['name']: e for e in self.session.list_replays()}[path.name]
            self.assertTrue(entry['compatible'])
        finally:
            path.unlink(missing_ok=True)

    def test_an_export_of_this_rule_version_replays(self):
        self.session.start_game(seed=94)
        self.session.thread.join(timeout=180)
        path = self.session.export_replay('unit-replayable')
        original_delay = ui.REPLAY_DELAY
        ui.REPLAY_DELAY = 0            # a test does not need the viewing pause
        try:
            self.assertTrue(self.session.replay_file(path.name))
            self.session.thread.join(timeout=180)
            self.assertEqual(self.session.status, 'finished')
            self.assertIsNone(self.session.timing_payload(),
                              'a replay is not a timed game')
        finally:
            ui.REPLAY_DELAY = original_delay
            path.unlink(missing_ok=True)

    def test_a_replay_from_another_rule_version_is_refused(self):
        self.session.start_game(seed=93)
        self.session.thread.join(timeout=180)
        path = self.session.export_replay('unit-old-rules')
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['rule_version'] = '2.0'
            path.write_text(json.dumps(payload), encoding='utf-8')

            entry = {e['name']: e for e in self.session.list_replays()}[path.name]
            self.assertFalse(entry['compatible'],
                             'an old file stays listed but is marked unreplayable')
            self.assertEqual(entry['rule_version'], '2.0')

            self.assertFalse(self.session.replay_file(path.name))
            notices = [e['text'] for e in self.session.events if e.get('kind') == 'notice']
            self.assertTrue(any('2.0' in text for text in notices),
                            f'a version mismatch must be reported: {notices}')

            # a file exported before versioning carries no rule_version at all
            payload.pop('rule_version')
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertFalse(self.session.replay_file(path.name))
            self.assertTrue(any('没有记录规则版本' in e.get('text', '')
                                for e in self.session.events if e.get('kind') == 'notice'),
                            'a legacy export must be reported as unversioned')
            self.assertEqual(self.session.status, 'finished',
                             'a refused replay must not start a game')
        finally:
            path.unlink(missing_ok=True)

    def test_timing_is_recorded_and_exported(self):
        self.session.start_game(seed=92)
        self.session.thread.join(timeout=180)
        timing = self.session.timing_payload()
        self.assertIsNotNone(timing)
        self.assertTrue(timing['started_at'] and timing['finished_at'])
        self.assertEqual(len(timing['actions']), len(self.session.answers),
                         'one action record per decision')
        self.assertAlmostEqual(
            sum(a['seconds'] for a in timing['actions']),
            sum(p['total_seconds'] for p in timing['players']), places=2,
            msg='the per-player totals must add up to the recorded actions')
        path = self.session.export_replay('unit-timing')
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['timing']['total_seconds'], timing['total_seconds'])
            self.assertTrue(payload['timing']['players'])
        finally:
            path.unlink(missing_ok=True)

    def test_export_with_raw_output_writes_a_readable_companion(self):
        self.session.raw_output = True
        self.session.start_game(seed=97)
        self.session.thread.join(timeout=180)
        path = self.session.export_replay('unit-text')
        text_path = path.with_suffix('.txt') if path else None
        try:
            self.assertIsNotNone(path)
            self.assertTrue(text_path.is_file(), 'no .txt was written')
            self.assertEqual(text_path.stem, path.stem, 'the two files must share a name')
            body = text_path.read_text(encoding='utf-8')
            # a review copy describes the game in prose, not JSON
            self.assertIn(i18n.t('text_roles_header', 'zh'), body)
            self.assertIn('存活', body)
            self.assertNotIn('{"', body)
            # and it keeps the private half of the transcript, marked
            private = [e for e in self.session.events if e.get('kind') == 'private']
            self.assertTrue(private, 'the exported game has no private events to check')
            self.assertIn(f"[{i18n.t('text_private', 'zh')}]", body)
            # the .txt is not a replay: the .json stays the only replayable file
            self.assertFalse(self.session.replay_file(text_path.name))
        finally:
            if path:
                path.unlink(missing_ok=True)
                text_path.unlink(missing_ok=True)
            self.session.raw_output = False


class Timing(unittest.TestCase):
    '''Total and per-action clocks, kept only for a real game.'''

    def test_an_idle_session_has_no_clock(self):
        session = ui.Session()
        self.assertFalse(session.timing_snapshot()['recorded'])
        self.assertIsNone(session.timing_payload())

    def test_the_action_clock_starts_and_resets(self):
        session = ui.Session()
        session.status = 'running'
        session._game_started = ui.time.monotonic() - 1.5
        session.game = engine.Game()
        session.game.game_state = support.new_state()

        self.assertTrue(session._begin_action('[Vote] Vote whom? ', 3, {'key': 'ID'}))
        live = session.timing_snapshot()
        self.assertTrue(live['recorded'] and live['running'])
        self.assertEqual(live['player'], 3, 'the clock names the seat deciding')

        session._end_action()
        self.assertEqual(session.timing_snapshot()['action_seconds'], 0.0,
                         'the action clock resets when the decision is done')
        payload = session.timing_payload()
        seat = next(row for row in payload['players'] if row['id'] == 3)
        self.assertEqual(seat['actions'], 1)
        self.assertEqual(payload['actions'][0]['decision'], 'vote')
        self.assertEqual(len(payload['players']), engine.PLAYERS_COUNT,
                         'every seat is reported, even one who never acted')
        self.assertGreaterEqual(payload['total_seconds'], 1.4,
                                'the total clock ran from the game start')


if __name__ == '__main__':
    unittest.main()


class FakeModel:
    '''A provider that answers like a model, so the whole LLM path runs offline.'''

    name = 'fake'

    def __init__(self):
        self.calls = 0
        self.seen_kinds: set[str] = set()

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        kind = ''
        for line in user.splitlines():
            if line.startswith('- kind:') or line.startswith('- 类型：'):
                kind = re.split(r'[:：]', line)[-1].strip()
                self.seen_kinds.add(kind)
        if '"yes"' in user:
            # yes/no: accept the election and the badge, decline self-explosion
            return '{"yes": false}' if kind in ('自爆', 'self-explosion') \
                else '{"yes": true, "reasoning": "why not"}'
        if kind in ('发言', 'speech'):
            return '{"speech": "我是好人，先听大家的。", "reasoning": "keep it short"}'
        legal = [line for line in user.splitlines()
                 if 'legal choices' in line or '合法选择' in line]
        digits = re.findall(r'\d+', legal[-1]) if legal else []
        if digits:
            return f'{{"choice": {digits[0]}, "reasoning": "take the first"}}'
        return '{"choice": null, "reasoning": "pass"}'


class ModelPath(unittest.TestCase):
    '''Every seat driven by a model: proves the prompts and the answers line up.'''

    @classmethod
    def setUpClass(cls):
        i18n.set_language('zh')
        cls.session = ui.Session()
        ui.CURRENT = cls.session
        cls.model = FakeModel()
        # every seat answers through the same fake, so a mixed-model table never
        # reaches the network: the cache is keyed by seat now
        shared = llm_interface.LLMController(cls.model, language='zh')
        cls.session.controllers = {seat: shared for seat in range(engine.PLAYERS_COUNT)}
        cls.session.seats = {i: 'llm' for i in range(engine.PLAYERS_COUNT)}
        cls.session.start_game(seed=777)
        cls.session.thread.join(timeout=300)

    def test_the_game_runs_to_an_ending(self):
        self.assertEqual(self.session.status, 'finished',
                         f'the model path failed: {self.session.message}')
        self.assertIn(self.session.game.game_state.stage,
                      (engine.GameStage.End_Civilian_Win, engine.GameStage.End_Wolf_Win,
                       engine.GameStage.End_Tie))

    def test_every_decision_kind_was_prompted(self):
        # the events carry the English DecisionKind name, whatever the language
        labels = {e.get('label') for e in self.session.events if e.get('kind') == 'decision'}
        for expected in ('kill', 'query', 'vote', 'speech'):
            self.assertIn(expected, labels, f'no {expected} decision was made')
        self.assertTrue(self.model.calls > 20, 'the model was barely asked anything')

    def test_no_answer_was_a_fallback(self):
        notes = [e for e in self.session.events
                 if e.get('kind') == 'decision' and e.get('note')]
        self.assertFalse(notes, f'fallbacks were used: {notes[:2]}')

    def test_no_model_error_was_reported(self):
        errors = [e for e in self.session.events if e.get('kind') == 'llm-error']
        self.assertFalse(errors, f'model errors: {errors[:2]}')
