import argparse
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def run(metadata, timeout):
    collector = json.loads(metadata.read_text())
    url = collector['url']
    parts = urlsplit(url)
    if parts.scheme != 'http' or parts.hostname not in ['127.0.0.1', 'localhost']:
        raise ValueError('A loopback benchmark URL is required')
    output = Path(collector['output'])
    if output.exists():
        raise ValueError('The benchmark already has a final report; refusing to rerun')
    with urlopen(url, timeout=5) as response:
        if response.status != 200:
            raise ValueError('Benchmark collector is unavailable')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    base = f'http://127.0.0.1:{port}'

    def request(method, path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        try:
            with urlopen(Request(base+path, payload, {'Content-Type': 'application/json'}, method=method), timeout=30) as response:
                result = json.load(response)
        except HTTPError as error:
            try:
                raise RuntimeError(error.read().decode()) from None
            finally:
                error.close()
        value = result.get('value', result)
        if isinstance(value, dict) and value.get('error'):
            raise RuntimeError(value.get('message', value['error']))
        return value

    session = None
    with tempfile.TemporaryFile() as log:
        driver = subprocess.Popen(['/usr/bin/safaridriver', '--port', str(port)], stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+10
            while True:
                try:
                    request('GET', '/status')
                    break
                except URLError:
                    if driver.poll() is not None or time.monotonic() >= deadline:
                        log.seek(0)
                        raise RuntimeError('SafariDriver did not start: '+log.read().decode()) from None
                    time.sleep(0.2)
            result = request('POST', '/session', {'capabilities': {'alwaysMatch': {'browserName': 'safari'}}})
            session = result['sessionId']
            request('POST', f'/session/{session}/url', {'url': url})
            print(json.dumps({'started': True, 'title': request('GET', f'/session/{session}/title')}), flush=True)
            deadline = time.monotonic()+timeout
            previous = -1
            while True:
                if output.exists():
                    try:
                        report = json.loads(output.read_text())
                        break
                    except json.JSONDecodeError:

                        pass
                checkpoints = sorted(output.parent.glob(output.stem+'.phase-*.json'))
                if len(checkpoints) != previous:
                    previous = len(checkpoints)
                    print(json.dumps({'completed_phase_reports': previous}), flush=True)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Benchmark did not finish before the deadline')
                if driver.poll() is not None:
                    raise RuntimeError('SafariDriver stopped before the report arrived')
                time.sleep(1)
            print(json.dumps({'saved': str(output), 'completed': report['completed'], 'error': report.get('error')}), flush=True)
        finally:
            if session:
                try:
                    request('DELETE', f'/session/{session}')
                except (OSError, RuntimeError):
                    pass
            if driver.poll() is None:
                driver.terminate()
                driver.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collector_metadata', type=Path)
    parser.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    try:
        run(args.collector_metadata, args.timeout)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Safari automation failed: {error}\n')
