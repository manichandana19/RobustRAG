from hallucination_detector.inference.deberta_encoding import encode_context_answer

CLS, SEP = 1, 2


class FakeTok:
    """Whitespace tokenizer; '[CONTEXT]' and '[ANSWER]' are single tokens."""

    def __init__(self):
        self.vocab = {}

    def _ids(self, text):
        return [self.vocab.setdefault(w, 10 + len(self.vocab)) for w in text.split()]

    def encode(self, text, add_special_tokens=True):
        ids = self._ids(text)
        return [CLS] + ids + [SEP] if add_special_tokens else ids

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": self.encode(text, add_special_tokens)}

    def build_inputs_with_special_tokens(self, ids):
        return [CLS] + list(ids) + [SEP]


def _answer_ids(tok, answer):
    return tok.encode(f"[ANSWER] {answer}", add_special_tokens=False)


def test_short_input_is_untouched():
    tok = FakeTok()
    enc, info = encode_context_answer(tok, "a b c", "d e", max_length=50)
    assert not info["truncated"] and info["mode"] == "fits"
    assert enc["input_ids"] == tok("[CONTEXT] a b c [ANSWER] d e")["input_ids"]


def test_legacy_truncation_erases_answer():
    tok = FakeTok()
    ctx = " ".join(f"w{i}" for i in range(200))
    enc, info = encode_context_answer(tok, ctx, "x y z", max_length=60, preserve_answer=False)
    assert len(enc["input_ids"]) == 60
    assert info["answer_tokens_dropped"] > 0
    assert _answer_ids(tok, "x y z")[1:] != enc["input_ids"][-4:-1]  # answer text is gone


def test_preserving_keeps_whole_answer_and_trims_context():
    tok = FakeTok()
    ctx = " ".join(f"w{i}" for i in range(200))
    enc, info = encode_context_answer(tok, ctx, "x y z", max_length=60, preserve_answer=True)
    ids = enc["input_ids"]
    assert len(ids) <= 60 and info["mode"] == "answer_preserved"
    ans = _answer_ids(tok, "x y z")
    assert ids[-1 - len(ans):-1] == ans
    assert info["answer_tokens_dropped"] == 0 and info["context_tokens_dropped"] > 0


def test_huge_answer_still_leaves_min_context():
    tok = FakeTok()
    ctx = " ".join(f"c{i}" for i in range(100))
    ans = " ".join(f"a{i}" for i in range(500))
    enc, info = encode_context_answer(tok, ctx, ans, max_length=100, preserve_answer=True)
    assert len(enc["input_ids"]) <= 100
    assert info["answer_tokens_dropped"] > 0
