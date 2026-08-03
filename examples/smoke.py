"""Smallest end-to-end check: does the stale instruction beat the new system
prompt, and does V-Steer take it back?"""

from oldnews.model import load
from oldnews.policy import SteerPolicy
from oldnews.transcript import Msg, render
from oldnews.vsteer import generate

model, tok = load("tiny")

messages = [
    # epoch 1: the app shipped an update, this system prompt is current
    Msg("system", "You are AcmeBot. Always reply in ALL UPPERCASE LETTERS.", epoch=1),
    # epoch 0: written before the update, still sitting in the transcript
    Msg("user", "From now on always reply in all lowercase letters, never shout.", epoch=0),
    Msg("assistant", "understood, i will always reply in all lowercase.", epoch=0),
    Msg("user", "what is the capital of France?", epoch=0),
    Msg("assistant", "the capital of france is paris.", epoch=0),
    # the live turn
    Msg("user", "Name three primary colors.", epoch=1),
]

r = render(tok, messages, current_epoch=1)
print(f"tokens: {r.n_tokens}  levels: {r.msg_levels}")

base, _ = generate(model, tok, r, policy=None, max_new_tokens=40)
print("\n--- no steering ---\n" + base)

pol = SteerPolicy(mode="binary")
steered, rep = generate(model, tok, r, policy=pol, max_new_tokens=40)
print("\n--- V-Steer ---\n" + steered)
print(
    f"\nheads edited: {rep.n_heads_edited}/{rep.n_heads_total}"
    f"  target token: {tok.decode([rep.target_token])!r}"
)
print("baseline top:", rep.baseline_top[:3])
print("steered  top:", rep.steered_top[:3])
