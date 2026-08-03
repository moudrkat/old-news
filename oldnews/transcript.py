"""Chat transcripts that carry a priority level per message.

The paper assumes two contiguous spans in one prompt: a privileged span A
(system) and a conflicting lower-priority span B (user). A production chat is
messier -- it is a list of messages, some of which were written under an older
version of the app. This module renders such a transcript through the model's
chat template and reports, for every token position, which priority level it
belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy import Priority


@dataclass
class Msg:
    """One chat message plus the metadata the hierarchy needs.

    ``epoch`` is the version of the app/system prompt this message was written
    under. Bump it whenever you ship a system-prompt change; messages left
    behind at a lower epoch are the "old news" this repo is about.
    """

    role: str
    content: str
    epoch: int = 0
    priority: Priority | None = None  # explicit override
    pinned: bool = False  # never demoted, whatever its epoch

    def to_chat(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Rendered:
    """A transcript laid out over token positions."""

    text: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]
    levels: list[int | None]  # per token: Priority value, or None if unassigned
    msg_spans: list[tuple[int, int]]  # per message: [start, end) token indices
    msg_levels: list[int]
    messages: list[Msg] = field(default_factory=list)

    def positions(self, *levels: int) -> list[int]:
        """Token indices belonging to any of ``levels``."""
        want = set(levels)
        return [i for i, lv in enumerate(self.levels) if lv in want]

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


def assign_priorities(messages: list[Msg], current_epoch: int | None = None) -> list[int]:
    """Default priority ladder for a production transcript.

    system                        -> SYSTEM   (privileged)
    last user message             -> USER
    tool output                   -> TOOL
    history at the current epoch  -> HISTORY
    history from an older epoch   -> STALE    (demoted)

    An explicit ``Msg.priority`` always wins; ``pinned`` blocks demotion.
    """
    if current_epoch is None:
        current_epoch = max((m.epoch for m in messages), default=0)

    last_user = max(
        (i for i, m in enumerate(messages) if m.role == "user"), default=None
    )

    out: list[int] = []
    for i, m in enumerate(messages):
        if m.priority is not None:
            out.append(int(m.priority))
            continue
        if m.role == "system":
            lv = Priority.SYSTEM if m.epoch >= current_epoch else Priority.STALE
        elif m.role == "tool":
            lv = Priority.TOOL
        elif i == last_user:
            lv = Priority.USER
        elif m.epoch < current_epoch:
            lv = Priority.STALE
        else:
            lv = Priority.HISTORY
        if m.pinned and lv == Priority.STALE:
            lv = Priority.HISTORY
        out.append(int(lv))
    return out


def render(
    tokenizer,
    messages: list[Msg],
    current_epoch: int | None = None,
    add_generation_prompt: bool = True,
    content_only: bool = True,
) -> Rendered:
    """Apply the chat template and map every token to a priority level.

    ``content_only=True`` labels only the message *body*, leaving role headers
    and template scaffolding unassigned (level ``None``, multiplier 1.0). That
    is the paper's V-Simple span strategy: whole message content, no extraction.
    """
    msg_levels = assign_priorities(messages, current_epoch)
    text = tokenizer.apply_chat_template(
        [m.to_chat() for m in messages],
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]

    levels: list[int | None] = [None] * len(input_ids)
    msg_spans: list[tuple[int, int]] = []

    cursor = 0
    for m, lv in zip(messages, msg_levels):
        needle = m.content if content_only else m.content
        start = text.find(needle, cursor)
        if start < 0:  # template mangled the content (rare); skip labelling it
            msg_spans.append((0, 0))
            continue
        end = start + len(needle)
        cursor = end
        tok_idx = [
            i
            for i, (a, b) in enumerate(offsets)
            if b > start and a < end and b > a
        ]
        if tok_idx:
            for i in tok_idx:
                levels[i] = lv
            msg_spans.append((tok_idx[0], tok_idx[-1] + 1))
        else:
            msg_spans.append((0, 0))

    return Rendered(
        text=text,
        input_ids=input_ids,
        offsets=offsets,
        levels=levels,
        msg_spans=msg_spans,
        msg_levels=msg_levels,
        messages=list(messages),
    )


def mark_constraint_spans(
    rendered: Rendered, patterns: dict[str, list[str]]
) -> Rendered:
    """Narrow labelling from whole messages to constraint substrings.

    ``patterns`` maps a priority level name to substrings to locate, e.g.
    ``{"SYSTEM": ["always answer in French"], "STALE": ["answer in English"]}``.
    This is the paper's tighter V-Steer span strategy (~5-20 tokens each);
    it usually needs fewer heads touched to get the same effect.
    """
    levels: list[int | None] = [None] * len(rendered.input_ids)
    for name, needles in patterns.items():
        lv = int(Priority[name])
        for needle in needles:
            start = rendered.text.find(needle)
            while start >= 0:
                end = start + len(needle)
                for i, (a, b) in enumerate(rendered.offsets):
                    if b > start and a < end and b > a:
                        levels[i] = lv
                start = rendered.text.find(needle, end)
    rendered.levels = levels
    return rendered
