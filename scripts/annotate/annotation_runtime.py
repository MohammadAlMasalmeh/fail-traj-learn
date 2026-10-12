"""Shared annotation IO, credentials, and HTTP transport without automatic retries."""
import hashlib
import json
import os
from pathlib import Path
import re
import uuid
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "gpt6_v3_12_episode_review_pack/episodes"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def redact(text, keys):
    for key in keys.values():
        text = text.replace(key, '[REDACTED]')
    return re.sub(r'(?<![A-Za-z0-9])(?:sk-[A-Za-z0-9_-]{24,}|AIza[A-Za-z0-9_-]{24,})', '[REDACTED]', text)

def post(url, provider, payload, keys):
    auth = {'Authorization': 'Bearer ' + keys[provider]} if provider == 'openai' else {'x-goog-api-key': keys[provider]}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json', **auth}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        try:
            error = json.loads(e.read()).get('error', {})
            message = str(error.get('message', ''))
        except Exception:
            message = 'Unparseable error body'
        raise RuntimeError(redact(f'{provider} HTTP {e.code}: {message}', keys)[:800]) from None
    except Exception as e:
        raise RuntimeError(f'{provider} {type(e).__name__}; no retry') from None

def usage_cost(provider, raw):
    if provider == 'openai':
        u = raw['usage']
        inp, out = u['input_tokens'], u['output_tokens']
        details = u.get('input_tokens_details') or {}
        cached, writes = details.get('cached_tokens', 0), details.get('cache_write_tokens', 0)
        thoughts = (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)
        cost = ((inp - cached - writes) * 2 + cached * .1 + writes * 2.5 + out * 10) / 1e6
    else:
        u = raw['usageMetadata']
        inp = u['promptTokenCount']
        cached, writes, thoughts = u.get('cachedContentTokenCount', 0), 0, u.get('thoughtsTokenCount', 0)
        out = u.get('candidatesTokenCount', 0) + thoughts
        cost = ((inp - cached) + cached * .1 + out * 5) / 1e6
    if any(type(x) is not int or x < 0 for x in (inp, out, cached, writes, thoughts)) or cached + writes > inp or thoughts > out:
        raise ValueError('Unexpected usage accounting')
    return {'input_tokens': inp, 'output_tokens_including_thinking': out, 'thinking_tokens': thoughts,
            'cached_tokens': cached, 'cache_write_tokens': writes,
            'generation_estimate_usd': round(cost, 8), 'usage': u}

def extract_answer(provider, raw):
    if provider == 'openai':
        text = ''.join(c.get('text', '') for item in raw.get('output', []) for c in item.get('content', []) if c.get('type') == 'output_text')
        return text, raw.get('status') == 'completed'
    candidates = raw.get('candidates', [])
    text = ''.join(p.get('text', '') for c in candidates for p in c.get('content', {}).get('parts', []) if not p.get('thought'))
    return text, len(candidates) == 1 and candidates[0].get('finishReason') == 'STOP'

def credentials():
    values = {"openai": os.environ.get("OPENAI_API_KEY", ""),
              "gemini": os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))}
    env = ROOT / ".env"
    legacy = []
    if env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip().removeprefix("export ")
            if not line or line.startswith("#"):
                continue
            name, sep, value = line.partition("=")
            value = (value if sep else name).split(" #", 1)[0].strip().strip("\"'")
            provider = {"OPENAI_API_KEY": "openai", "GEMINI_API_KEY": "gemini",
                        "GOOGLE_API_KEY": "gemini"}.get(name.strip()) if sep else None
            if provider:
                values[provider] = values[provider] or value
            elif not sep:
                legacy.append(value)
        if len(legacy) == 2:
            for provider, value in zip(("openai", "gemini"), legacy):
                values[provider] = values[provider] or value
    if not values["openai"]:
        raise ValueError("Set OPENAI_API_KEY in the environment or .env")
    return {k: v for k, v in values.items() if v}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        write_json(temp, value)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
