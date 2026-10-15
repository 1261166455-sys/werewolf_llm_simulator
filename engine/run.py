'''
Launch the werewolf operator console.

    ./venv/bin/python run.py

Loads the model API key from ds_key.txt, picks a free port, starts the web
console and opens the page in the default browser.

The key is only ever placed in this process's environment: it is never written
to disk, never logged, and never sent to the browser (the console only learns
whether a key exists).
'''
from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

import i18n
import ui

KEY_FILE = Path(__file__).resolve().parent / 'ds_key.txt'

# names accepted in ds_key.txt -> the environment variable they set
SETTING_ENV = {
    'kind': 'LLM_KIND',
    'model': 'LLM_MODEL',
    'base_url': 'LLM_BASE_URL',
}
ASSIGNMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+)$')


def read_key_file(path: Path) -> dict[str, str]:
    '''
    Parse the key file.

    A bare key on its own is the common case, but `name = value` lines are also
    accepted so the provider can be configured in the same file:

        sk-abc123...                     # just the key
        # or
        api_key = sk-abc123...
        model   = deepseek-chat
        base_url = https://api.deepseek.com/v1

    Blank lines and #comments are ignored. Any unrecognised `name = value` line
    whose name looks like a key is still taken as the key, because that is what
    it almost certainly is.
    '''
    settings: dict[str, str] = {}
    if not path.exists():
        return settings

    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = ASSIGNMENT.match(line)
        if not match:
            settings.setdefault('api_key', line)
            continue
        name, value = match.group(1).lower(), match.group(2).strip().strip('\'"')
        if name in SETTING_ENV:
            settings[name] = value
        elif 'key' in name or name in {'token', 'secret'}:
            settings['api_key'] = value
        else:
            settings.setdefault('api_key', value)
    return settings


def connect_host(host: str) -> str:
    '''A wildcard bind address is not something a browser can visit.'''
    return '127.0.0.1' if host in ('', '0.0.0.0', '::') else host


def free_port(preferred: int, host: str, tries: int = 20) -> int:
    '''
    First free port at or after `preferred`.

    Starting a second console while the first is still running would otherwise
    die with "address already in use", which is a confusing way to fail.
    '''
    bind_host = connect_host(host)
    for port in range(preferred, preferred + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # Match what uvicorn can actually do: SO_REUSEADDR allows binding a
            # socket left in TIME_WAIT by a previous run, but still refuses a
            # port that another process is actively listening on.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((bind_host, port))
            except OSError:
                continue
            return port
    raise SystemExit(f'no free port in {preferred}-{preferred + tries - 1}')


def open_when_ready(url: str, host: str, port: int, timeout: float = 30.0) -> bool:
    '''Open the browser only once the server accepts connections.'''
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return True
        except OSError:
            time.sleep(0.15)
    print(f'! no response from {url}; open it manually', file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Launch the werewolf operator console.')
    parser.add_argument('--key-file', type=Path, default=KEY_FILE,
                        help='file holding the API key (default: ds_key.txt)')
    parser.add_argument('--host', default=ui.HOST)
    parser.add_argument('--port', type=int, default=ui.PORT)
    parser.add_argument('--no-browser', action='store_true',
                        help='do not open the browser')
    parser.add_argument('--lang', default=i18n.DEFAULT, choices=sorted(i18n.LANGUAGES),
                        help='language for the console and the model prompt')
    args = parser.parse_args(argv)

    # set before the session is built: it reads the language and the key
    i18n.set_language(args.lang)

    settings = read_key_file(args.key_file)
    key = settings.pop('api_key', '')
    for name, value in settings.items():
        os.environ.setdefault(SETTING_ENV[name], value)
    if key:
        os.environ['LLM_API_KEY'] = key
        os.environ.setdefault('DEEPSEEK_API_KEY', key)

    # The session reads provider settings from the environment, so the key has
    # to be in place before it is constructed.
    ui.CURRENT = ui.Session()
    config = ui.CURRENT.provider_config

    host = connect_host(args.host)
    port = free_port(args.port, args.host)
    url = f'http://{host}:{port}/'

    lang = ui.CURRENT.language
    model_seats = sum(1 for v in ui.CURRENT.seats.values() if v == 'llm')
    human_seats = sum(1 for v in ui.CURRENT.seats.values() if v == 'human')
    print(i18n.t('banner_title', lang))
    print(i18n.t('banner_seats', lang, model=model_seats, human=human_seats))
    print(i18n.t('banner_model', lang, kind=config.kind, model=config.model))
    print(i18n.t('banner_endpoint', lang, url=config.base_url))
    print(i18n.t('banner_key_loaded', lang, file=args.key_file.name) if config.api_key
          else i18n.t('banner_key_missing', lang))
    print(i18n.t('banner_page', lang, url=url), flush=True)
    if not config.api_key:
        print(i18n.t('key_missing_hint', lang, file=args.key_file), file=sys.stderr)
    if port != args.port:
        print(i18n.t('port_busy', lang, port=args.port, used=port))
    # Flush now: stdout is block-buffered when it is not a terminal, so without
    # this the banner sits invisible behind uvicorn's log lines.
    sys.stdout.flush()

    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url, host, port),
                         name='browser', daemon=True).start()

    try:
        uvicorn.run(ui.app, host=args.host, port=port, log_level='info')
    except KeyboardInterrupt:
        pass
    finally:
        print('\n' + ('控制台已停止' if ui.CURRENT.language == 'zh' else 'console stopped'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
