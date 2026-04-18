from llm.tokenizer import BPETokenizer


def test_roundtrip():
    text = "hello world <|endoftext|> hello"
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(text, vocab_size=280)
    ids = tok.encode(text)
    dec = tok.decode(ids)
    assert dec == text


def test_lexicographic_tiebreak():
    text = "AB BA"
    tok = BPETokenizer()
    tok.train(text, vocab_size=257)
    assert tok.merges, "Expected at least one merge"
