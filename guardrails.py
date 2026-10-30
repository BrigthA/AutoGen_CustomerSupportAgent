"""Deterministic baseline guardrails. Not a substitute for auth, moderation or DLP."""
import re
from dataclasses import dataclass

MAX_QUERY_CHARS = 3000
MAX_LINES = 60

@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str = ''
    message: str = ''

# Detect explicit attempts to override assistant policy or obtain secrets.
INJECTION_PATTERNS = [
    r'ignore\s+(?:all\s+)?(?:previous|prior|system|developer)\s+(?:instructions|rules|prompts)',
    r'(?:reveal|print|show|dump|expose)\s+(?:your\s+)?(?:system\s+prompt|developer\s+instructions|api\s+key|secret\s+key)',
    r'(?:bypass|disable|turn\s+off)\s+(?:the\s+)?(?:guardrails|safety\s+checks|security\s+policy)',
    r'(?:you\s+are\s+now|act\s+as)\s+(?:an?\s+)?(?:unrestricted|jailbroken)\s+(?:assistant|model)',
]

SECRET_PATTERNS = [
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\bsk-[A-Za-z0-9_-]{20,}\b',
    r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
]


def check_input(query: str) -> GuardrailResult:
    if not isinstance(query, str) or not query.strip():
        return GuardrailResult(False, 'empty', 'Please enter a question.')
    if len(query) > MAX_QUERY_CHARS or len(query.splitlines()) > MAX_LINES:
        return GuardrailResult(False, 'length', f'Please shorten your question to {MAX_QUERY_CHARS} characters or fewer.')
    if any(ord(c) < 32 and c not in '\n\r\t' for c in query):
        return GuardrailResult(False, 'control_characters', 'Please remove unsupported control characters.')
    if any(re.search(p, query, flags=re.I) for p in SECRET_PATTERNS):
        return GuardrailResult(False, 'secret', 'Your message appears to contain a credential or private key. Remove it before submitting.')
    if any(re.search(p, query, flags=re.I) for p in INJECTION_PATTERNS):
        return GuardrailResult(False, 'prompt_injection', 'I cannot follow instructions to bypass safeguards or disclose internal instructions. Please ask a support question.')
    return GuardrailResult(True)


def check_output(answer: str) -> GuardrailResult:
    if not isinstance(answer, str) or not answer.strip():
        return GuardrailResult(False, 'empty_output', 'I could not generate a verified response. Please contact support.')
    if any(re.search(p, answer, flags=re.I) for p in SECRET_PATTERNS):
        return GuardrailResult(False, 'secret_output', 'The response requires human review before it can be shown.')
    return GuardrailResult(True)
