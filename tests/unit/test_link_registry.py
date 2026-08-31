"""The user's links: captured once, numbered stably, distinguishable, repairable.

The failure these cover is a link that reached the orchestrator and never
reached the worker, or reached it as the *other* link from the same message.
So the cases are: two links in one request stay two links with the right
sentence attached to each; re-running the capture does not renumber them; the
table lands in the prompt of every agent, not just the one that read the
message; and a URL the model retyped short is put back the way the user wrote it.
"""
import pytest

from CoScientist.agents.callbacks.link_registry import (
    LINKS_CONTEXT_STATE_KEY,
    USER_LINKS_STATE_KEY,
    classify_url,
    expand_link_refs,
    expand_refs,
    find_urls,
    normalize_url,
    redact_link_urls,
    register_tool_result_links,
    link_id_for,
    link_ref,
    register_user_links,
    render_links_block,
    resolve_link_refs,
    user_links,
)

TRAIN = "https://data.example.org/sets/train.zip?sig=AAA111&exp=99"
TEST = "https://data.example.org/sets/test.zip?sig=BBB222&exp=99"
HALLUCINATED = "https://hallucinated.example/nope.zip"

# Ids are digests of the url, so the test derives them the same way the code
# does rather than hardcoding them — that is the property under test.
ID_TRAIN, ID_TEST = link_id_for(TRAIN), link_id_for(TEST)
REF_TRAIN, REF_TEST = link_ref(ID_TRAIN), link_ref(ID_TEST)
# Syntactically valid, deliberately never registered.
REF_UNKNOWN = "[[link0000]]"


class _Ctx:
    """Stand-in for ADK's CallbackContext / ToolContext (state + user_content)."""

    def __init__(self, text=None, state=None):
        self.state = {} if state is None else state
        self.user_content = _Content(text) if text is not None else None


class _Content:
    def __init__(self, text):
        self.parts = [_Part(text)]


class _Part:
    def __init__(self, text):
        self.text = text


class _Tool:
    name = "run_sandbox_task"


# ── extraction ───────────────────────────────────────────────────────────────
def test_prose_punctuation_is_not_swallowed_into_the_url():
    text = f"Возьми {TRAIN}, а тестовую — {TEST}."
    assert [u for u, _s, _e in find_urls(text)] == [TRAIN, TEST]


def test_markdown_link_closing_paren_is_dropped_but_wiki_parens_are_kept():
    md = "see [data](https://example.org/a.zip)"
    assert find_urls(md)[0][0] == "https://example.org/a.zip"
    wiki = "https://en.wikipedia.org/wiki/Ionic_liquid_(solvent)"
    assert find_urls(f"read {wiki} first")[0][0] == wiki


@pytest.mark.parametrize("code", [
    "import scipy.io",
    "self.net = nn.Linear(3, 4)",
    "df.info()",
    "df.at[0, 'x']",
    "bash run.sh --flag",
    "sklearn.linear_model.LogisticRegression",
    "numpy 1.26.4",
    "сохрани в train.csv",
    "напиши на bob@example.com",
    "objект.метод(1)",
    "сохрани выборка.csv и модель.pkl",
])
def test_a_dotted_code_token_is_not_a_link(code):
    """The price of reading a bare host as a link, and the bound on it.

    `host.tld` is also the shape of half the code this system passes around.
    Registering one of these would put a `[[linkN]]` where an import used to
    be, in every message the model then reads.
    """
    assert find_urls(code, bare_hosts=True) == []


def test_a_collision_prone_tld_counts_only_with_a_path():
    # `scipy.io` is code; the same shape with a path is unambiguously a link.
    assert find_urls("import scipy.io", bare_hosts=True) == []
    assert find_urls("возьми huggingface.co/datasets/imdb", bare_hosts=True)[0][0] == (
        "huggingface.co/datasets/imdb")


def test_a_cyrillic_domain_typed_by_hand_is_a_link():
    """The requests here are written in Russian, so the link can be too.

    Host labels are matched as Unicode letters rather than `[a-z0-9]`, which
    is the only thing that stood between an IDN host and the same treatment
    every other link gets.
    """
    assert find_urls("смотри пример.рф", bare_hosts=True)[0][0] == "пример.рф"
    assert find_urls("данные тут данные.рф/выборка.csv", bare_hosts=True)[0][0] == (
        "данные.рф/выборка.csv")
    # Case-folded, and the sentence's full stop is not part of the host.
    assert find_urls("открой ПРИМЕР.РФ.", bare_hosts=True)[0][0] == "ПРИМЕР.РФ"
    # A punycode TLD needs no list of its own: `-` cannot be in an identifier.
    assert find_urls("см. xn--e1afmkfd.xn--p1ai", bare_hosts=True)[0][0] == (
        "xn--e1afmkfd.xn--p1ai")


def test_a_cyrillic_domain_and_its_punycode_are_the_same_link():
    """One host the user typed and a tool echoed back must not become two
    entries the model has to choose between."""
    assert normalize_url("пример.рф") == normalize_url("XN--E1AFMKFD.XN--P1AI")
    assert link_id_for("пример.рф") == link_id_for("https://xn--e1afmkfd.xn--p1ai/")
    # The port survives the fold rather than being swallowed into a label.
    assert normalize_url("https://пример.рф:8080/a") == (
        "https://xn--e1afmkfd.xn--p1ai:8080/a")


def test_a_bucket_uri_a_tool_hands_back_is_registered_as_a_link():
    """`s3://bucket/key`, not a signed https url, is what the dataset-collection
    tool this system delegates to actually returns — a scheme the plain
    http(s)/www regex never matched, so the link skipped the registry
    silently rather than failing loudly.
    """
    uri = "s3://my-research-bucket/session/final_dataset.csv"
    assert find_urls(uri) == [(uri, 0, len(uri))]
    assert classify_url(uri) == ("data file", "final_dataset.csv")

    ctx = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, ctx,
                               {"metadata": {"dataset_s3_path": uri}})
    assert ctx.state[USER_LINKS_STATE_KEY][link_id_for(uri)]["url"] == uri


def test_extra_schemes_can_be_switched_off_without_a_code_change(monkeypatch):
    """The escape hatch: LINK_REGISTRY_EXTRA_SCHEMES=0 restores the exact
    pre-existing http(s)/www behaviour, for a same-day rollback if this
    collides with someone else's work on the sandbox/dataset tooling."""
    monkeypatch.setenv("LINK_REGISTRY_EXTRA_SCHEMES", "0")
    import importlib
    from CoScientist.agents.callbacks import link_registry as mod
    importlib.reload(mod)
    try:
        assert mod.find_urls("s3://bucket/key.csv") == []
        assert mod.find_urls("https://example.org/a.zip") != []
    finally:
        importlib.reload(mod)  # restore the default for every test after this one


def test_normalization_folds_cosmetics_but_never_the_signature():
    a = "HTTPS://WWW.Example.org/Sets/train.zip/?sig=AAA111#frag"
    b = "https://example.org/Sets/train.zip?sig=AAA111"
    assert normalize_url(a) == normalize_url(b)
    # Two presigned URLs differing only in the signature stay two links.
    assert normalize_url(TRAIN) != normalize_url(TEST)


@pytest.mark.parametrize("url,role,expected_label", [
    (TRAIN, "dataset archive", "train.zip"),
    ("https://example.org/x/results.csv", "data file", "results.csv"),
    ("https://arxiv.org/abs/2401.00001", "paper", "abs/2401.00001"),
    ("https://github.com/owner/repo", "code repository", "owner/repo"),
    ("https://example.org/docs", "web page", "example.org/docs"),
    ("https://storage.example.org/raw-binary?X-Amz-Signature=123", "file download", "raw-binary"),
    ("https://example.org/complex.pdb", "chemical/biological data", "complex.pdb"),
    ("https://example.org/model.pt", "model checkpoint", "model.pt"),
])
def test_role_is_derived_from_the_url_shape(url, role, expected_label):
    detected_role, label = classify_url(url)
    assert detected_role == role
    assert label == expected_label


# ── two links in one message ─────────────────────────────────────────────────
def test_two_links_get_distinct_ids_and_the_user_s_own_wording():
    state = {}
    register_user_links(
        state,
        f"Обучающая выборка лежит здесь: {TRAIN} , а тестовая — {TEST} . Обучи на первой.",
    )
    registry = state[USER_LINKS_STATE_KEY]

    assert [e["url"] for e in registry.values()] == [TRAIN, TEST]
    assert registry[ID_TRAIN]["label"] == "train.zip"
    assert registry[ID_TEST]["label"] == "test.zip"
    # Each mention carries the phrase that assigns the link its meaning, with the
    # sibling shown as its reference — that pairing makes them tellable apart.
    assert "Обучающая выборка" in registry[ID_TRAIN]["mention"]
    assert REF_TEST in registry[ID_TRAIN]["mention"]
    assert "тестовая" in registry[ID_TEST]["mention"]


def test_recapture_is_idempotent_so_ids_never_shift_under_the_agents():
    state = {}
    register_user_links(state, f"train {TRAIN} test {TEST}")
    before = {k: v["url"] for k, v in state[USER_LINKS_STATE_KEY].items()}

    register_user_links(state, f"как я писал, {TRAIN} — обучение")
    after = {k: v["url"] for k, v in state[USER_LINKS_STATE_KEY].items()}

    assert before == after


def test_a_later_turn_adds_links_without_rewriting_the_first_mention():
    state = {}
    register_user_links(state, f"обучающая — {TRAIN}")
    first_mention = state[USER_LINKS_STATE_KEY][ID_TRAIN]["mention"]

    register_user_links(state, f"ещё возьми {TEST}, а {TRAIN} уже был")
    registry = state[USER_LINKS_STATE_KEY]

    assert registry[ID_TEST]["url"] == TEST
    assert registry[ID_TRAIN]["mention"] == first_mention


def test_render_shows_id_role_and_mention_but_never_the_url():
    """The URL is withheld on purpose: nothing a model reads should contain a
    string it could copy instead of writing the reference."""
    state = {}
    register_user_links(state, f"обучение {TRAIN} и тест {TEST}")
    block = render_links_block(state[USER_LINKS_STATE_KEY])

    assert "(2)" in block
    for link_id in (ID_TRAIN, ID_TEST):
        assert f"**[[{link_id}]]**" in block
    assert TRAIN not in block
    assert TEST not in block
    # ...even though the registry itself still holds the real URLs — the
    # substitution machinery reads from there, not from the rendered block.
    assert state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN
    assert render_links_block({}) == ""


# ── the callback: capture at the root, render everywhere ─────────────────────
def test_root_captures_and_renders():
    ctx = _Ctx(text=f"обучи на {TRAIN}")
    user_links(ctx)

    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN
    assert REF_TRAIN in ctx.state[LINKS_CONTEXT_STATE_KEY]
    assert TRAIN not in ctx.state[LINKS_CONTEXT_STATE_KEY]


def test_the_scheme_the_user_did_not_type_does_not_cost_them_a_reference():
    """The whole round trip for a link written the way people actually write one.

    `example.com` used to match nothing, so the request kept its raw link all
    the way to the model — the exact leak the registry exists to close, and
    invisible precisely because everything downstream behaved correctly on an
    empty registry.
    """
    ctx = _Ctx(text="придумай query-параметр к ссылке example.com и передай дальше")
    user_links(ctx)
    ref = link_ref(link_id_for("example.com"))

    # Registered under the scheme code will hand on, described by its shape.
    assert ctx.state[USER_LINKS_STATE_KEY][link_id_for("example.com")]["url"] == (
        "https://example.com")
    assert ref in ctx.state[LINKS_CONTEXT_STATE_KEY]

    # The model sees the reference, not the link, in its own message...
    req = _Request(["придумай query-параметр к ссылке example.com и передай дальше"])
    redact_link_urls(callback_context=ctx, llm_request=req)
    assert req.text() == f"придумай query-параметр к ссылке {ref} и передай дальше"

    # ...and what leaves for the next agent is a real, schemed URL.
    args = {"request": f"добавь ?x=1 к {ref}"}
    resolve_link_refs(_Tool(), args, ctx)
    assert args["request"] == "добавь ?x=1 к https://example.com"


def test_a_context_block_is_not_read_as_loosely_as_the_message():
    """Only the human's own message is read loosely.

    Everything else in a turn's context was rendered from state by code, so a
    url in it always carries its scheme — and reading it loosely would only
    ever buy a false positive off a line of code stored as evidence.
    """
    ctx = _Ctx(text="продолжай", state={"research_context": "см. код: import scipy.io"})
    user_links(ctx)

    assert not ctx.state.get(USER_LINKS_STATE_KEY)
    assert ctx.state[LINKS_CONTEXT_STATE_KEY] == ""


def test_a_worker_renders_the_table_without_its_caller_having_repeated_the_link():
    root = _Ctx(text=f"обучи на {TRAIN} и проверь на {TEST}")
    user_links(root)

    # The worker's own message is the orchestrator's prose, with no URL in it.
    worker = _Ctx(text="Обучи модель на обучающей выборке.", state=root.state)
    user_links(worker)

    block = worker.state[LINKS_CONTEXT_STATE_KEY]
    assert REF_TRAIN in block and REF_TEST in block
    assert TRAIN not in block and TEST not in block


def test_a_url_a_caller_invented_gets_an_id_but_cannot_corrupt_a_real_one():
    """The accepted cost of re-extracting at every hop.

    A worker registers whatever URLs are in the text it was handed. Normally
    those were written by `resolve_link_refs`, but a caller can also mention a
    URL of its own — and that one gets an id too. The bound that makes this
    acceptable: registration is idempotent by normalised URL, so a newcomer is
    only ever ADDED. It cannot renumber, overwrite or displace a link that is
    already in the table.
    """
    worker = _Ctx(text=f"обучи на {TRAIN}", state={})
    user_links(worker)
    real = dict(worker.state[USER_LINKS_STATE_KEY][ID_TRAIN])

    worker.user_content = _Content(f"а ещё скачай {HALLUCINATED}")
    user_links(worker)
    registry = worker.state[USER_LINKS_STATE_KEY]

    assert registry[ID_TRAIN] == real         # the genuine link is untouched
    assert registry[link_id_for(HALLUCINATED)]["url"] == HALLUCINATED


def test_no_links_means_no_prompt_section_at_all():
    ctx = _Ctx(text="посчитай logP для аспирина")
    user_links(ctx)

    assert ctx.state[LINKS_CONTEXT_STATE_KEY] == ""


# ── before_tool: ids expand, clipped URLs are repaired ───────────────────────
def _registry_state():
    state = {}
    register_user_links(state, f"обучение {TRAIN} и тест {TEST}")
    return state


def test_a_link_id_argument_expands_to_the_real_url():
    ctx = _Ctx(state=_registry_state())
    args = {"task": "train a model", "dataset_url": REF_TEST}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["dataset_url"] == TEST
    assert args["task"] == "train a model"


def test_a_url_retyped_without_its_query_string_is_restored():
    ctx = _Ctx(state=_registry_state())
    args = {"dataset_url": TRAIN.split("?")[0]}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["dataset_url"] == TRAIN


def test_a_url_clipped_mid_signature_is_restored_inside_prose():
    ctx = _Ctx(state=_registry_state())
    clipped = TRAIN[:-4]
    args = {"request": f"Скачай {clipped} и обучи на нём."}
    resolve_link_refs(_Tool(), args, ctx)

    assert TRAIN in args["request"]


def test_a_path_prefix_is_left_alone_when_nothing_was_actually_dropped():
    state = {}
    register_user_links(state, "данные тут https://example.org/data/train.csv")
    ctx = _Ctx(state=state)
    args = {"path": "https://example.org/data"}
    resolve_link_refs(_Tool(), args, ctx)

    # A shorter path is a different resource, not a truncation to repair.
    assert args["path"] == "https://example.org/data"


def test_an_unrelated_url_and_an_unknown_id_pass_through_untouched():
    ctx = _Ctx(state=_registry_state())
    args = {"a": "https://other.example.org/thing.zip", "b": REF_UNKNOWN, "c": "Landau"}
    resolve_link_refs(_Tool(), args, ctx)

    assert args == {"a": "https://other.example.org/thing.zip", "b": REF_UNKNOWN, "c": "Landau"}


def test_an_empty_registry_short_circuits():
    ctx = _Ctx(state={})
    args = {"dataset_url": REF_TRAIN}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["dataset_url"] == REF_TRAIN


def test_one_shared_quote_replaces_identical_per_link_mentions():
    state = {}
    register_user_links(state, f"обучение {TRAIN} и тест {TEST}")
    block = render_links_block(state[USER_LINKS_STATE_KEY])

    # A short request gives every link the same window; it is quoted once.
    assert block.count("The request said") == 1
    assert "links shown as their references" in block


def test_distinct_mentions_are_kept_per_link_in_a_long_request():
    filler = "и вот ещё много подробностей про постановку задачи, " * 8
    state = {}
    register_user_links(state, f"обучающая выборка {TRAIN} {filler} а тестовая {TEST}")
    registry = state[USER_LINKS_STATE_KEY]
    block = render_links_block(registry)

    assert registry[ID_TRAIN]["mention"] != registry[ID_TEST]["mention"]
    assert block.count("mentioned as") == 2


def test_a_reference_inside_a_list_argument_is_expanded():
    """The shape that broke a real ResearchAgent call.

    `tavily_extract` takes `urls=[…]`, not a url. The reference sat one level
    down, the callback only looked at arguments that were themselves strings,
    and the tool was handed the literal `[[link…]]` — `Validation Error:
    Invalid URL format`. Nothing about that failure is specific to Tavily: any
    tool taking a list of links had the same hole.
    """
    ctx = _Ctx(state=_registry_state())
    args = {"urls": [REF_TRAIN, REF_TEST], "query": "GSK-3 selectivity"}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["urls"] == [TRAIN, TEST]
    assert args["query"] == "GSK-3 selectivity"


def test_a_reference_is_expanded_at_any_depth_and_other_types_are_left_alone():
    ctx = _Ctx(state=_registry_state())
    args = {"opts": {"target": {"url": REF_TRAIN}, "penalty": "l1"},
            "pair": (REF_TEST, "L2"), "limit": 5, "strict": True, "nothing": None}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["opts"] == {"target": {"url": TRAIN}, "penalty": "l1"}
    assert args["pair"] == (TEST, "L2")          # tuple stays a tuple
    assert (args["limit"], args["strict"], args["nothing"]) == (5, True, None)


def test_an_argument_with_nothing_to_resolve_is_not_rebuilt():
    """Unchanged containers come back as the SAME object, so the callback can
    tell a real rewrite from a copy without comparing values of unknown type."""
    ctx = _Ctx(state=_registry_state())
    urls, opts = ["https://unrelated.example.org/page"], {"penalty": "l1"}
    args = {"urls": urls, "opts": opts}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["urls"] is urls and args["opts"] is opts


def test_an_ml_hyperparameter_is_never_mistaken_for_a_reference():
    """`penalty="l1"` must not leave here as a presigned URL.

    Expansion now fires anywhere in a string, prose included, so the token
    itself has to carry the discrimination. `[[…]]` does; a bare `l1`/`L2`
    could not — the coder family passes those as regularisation settings on
    nearly every call.
    """
    ctx = _Ctx(state=_registry_state())
    args = {"penalty": "l1", "norm": "L2", "loss": "l2", "layer": "L1",
            "note": f"use L2 regularization with {ID_TRAIN} dropout"}
    resolve_link_refs(_Tool(), args, ctx)

    assert args == {"penalty": "l1", "norm": "L2", "loss": "l2", "layer": "L1",
                    "note": f"use L2 regularization with {ID_TRAIN} dropout"}


def test_a_reference_expands_anywhere_in_a_string_not_just_in_url_arguments():
    """The task text of a delegated step is where a link most needs to survive,
    and that argument is named `request`/`task`, not `url`."""
    ctx = _Ctx(state=_registry_state())
    args = {
        "request": f"Обучи на {REF_TRAIN}, проверь на {REF_TEST} и сравни.",
        "dataset_url": REF_TRAIN,
    }
    resolve_link_refs(_Tool(), args, ctx)

    assert args["request"] == f"Обучи на {TRAIN}, проверь на {TEST} и сравни."
    assert args["dataset_url"] == TRAIN


def test_a_reference_is_matched_leniently_on_case_and_spacing():
    ctx = _Ctx(state=_registry_state())
    args = {"request": f"возьми [[{ID_TRAIN.upper()}]] и [[ {ID_TEST} ]]"}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["request"] == f"возьми {TRAIN} и {TEST}"


def test_an_unknown_reference_is_left_visible_rather_than_dropped():
    # A dangling reference is a legible bug; a silently deleted one hands the
    # next agent a sentence with its object missing.
    ctx = _Ctx(state=_registry_state())
    args = {"request": f"скачай {REF_UNKNOWN}"}
    resolve_link_refs(_Tool(), args, ctx)

    assert args["request"] == f"скачай {REF_UNKNOWN}"


def test_a_clipped_url_is_still_repaired_alongside_reference_expansion():
    ctx = _Ctx(state=_registry_state())
    args = {"request": f"Скачай {TRAIN.split('?')[0]} и обучи на нём."}
    resolve_link_refs(_Tool(), args, ctx)

    assert TRAIN in args["request"]


# ── after_model: a reference must never reach a human ────────────────────────
class _Part:
    def __init__(self, text=None):
        self.text = text


class _Response:
    def __init__(self, texts):
        self.content = type("C", (), {"parts": [_Part(t) for t in texts]})()


def test_the_final_answer_never_shows_a_raw_reference():
    """`before_tool` covers tool calls, but a report, an output_key or the reply
    the user reads never passes through one."""
    ctx = _Ctx(state=_registry_state())
    response = _Response([f"Готово. Данные брал из {REF_TRAIN}, метрики — на {REF_TEST}."])
    expand_link_refs(callback_context=ctx, llm_response=response)

    text = response.content.parts[0].text
    assert text == f"Готово. Данные брал из {TRAIN}, метрики — на {TEST}."
    assert "[[" not in text


def test_an_answer_without_references_is_left_untouched():
    ctx = _Ctx(state=_registry_state())
    response = _Response(["Обучил модель, accuracy 0.91 при penalty=l1."])
    expand_link_refs(callback_context=ctx, llm_response=response)

    assert response.content.parts[0].text == "Обучил модель, accuracy 0.91 при penalty=l1."


def test_a_raw_retyped_url_with_no_reference_is_still_repaired_in_the_final_answer():
    """The gap this closes: a model that ignores its instructions and retypes
    the URL directly (no [[…]] at all) used to sail through unrepaired here,
    even though the SAME mistake in a tool argument was already caught by
    resolve_link_refs. expand_link_refs now runs the identical repair."""
    ctx = _Ctx(state=_registry_state())
    clipped = TRAIN.split("?")[0]   # the query string — where the signature lives — dropped
    response = _Response([f"Готово, датасет тут: {clipped}"])
    expand_link_refs(callback_context=ctx, llm_response=response)

    assert TRAIN in response.content.parts[0].text


def test_a_link_survives_coder_to_orchestrator_to_a_third_agent():
    """The user's scenario end to end: CoderAgent's sandbox hands back a fresh
    artifact URL, CoderAgent reports it to the orchestrator, and the
    orchestrator delegates it onward to a third agent — three independent
    local registries, one real URL, no state shared between any of them."""
    ARTIFACT = "https://minio.internal/bucket/model.pt?sig=CCC333"

    # 1) CoderAgent's sandbox call returns the artifact; the agent's own
    #    after_tool hook registers it mid-turn.
    coder = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, coder,
                               {"status": "done", "artifact_url": ARTIFACT})
    assert coder.state[USER_LINKS_STATE_KEY][link_id_for(ARTIFACT)]["url"] == ARTIFACT

    # 2) CoderAgent's final answer follows instructions and writes the
    #    reference; expand_link_refs turns it back into the real URL before
    #    the AgentTool call returns to the orchestrator.
    coder_answer = _Response([f"Модель обучена, чекпоинт: {link_ref(link_id_for(ARTIFACT))}."])
    expand_link_refs(callback_context=coder, llm_response=coder_answer)
    tool_result_text = coder_answer.content.parts[0].text
    assert tool_result_text == f"Модель обучена, чекпоинт: {ARTIFACT}."

    # 3) That text is what AgentTool hands back — the orchestrator's own
    #    after_tool hook re-extracts it into ITS OWN registry (a fresh id;
    #    ids are local, never shared between agents).
    orchestrator = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, orchestrator, tool_result_text)
    assert orchestrator.state[USER_LINKS_STATE_KEY][link_id_for(ARTIFACT)]["url"] == ARTIFACT
    assert link_ref(link_id_for(ARTIFACT)) in orchestrator.state[LINKS_CONTEXT_STATE_KEY]

    # 4) The orchestrator delegates onward by reference; resolve_link_refs
    #    expands it into the real URL in the outbound request — the third
    #    agent's incoming message therefore also contains the real URL.
    args = {"request": f"Приложи к отчёту чекпоинт {link_ref(link_id_for(ARTIFACT))}."}
    resolve_link_refs(_Tool(), args, orchestrator)
    assert args["request"] == f"Приложи к отчёту чекпоинт {ARTIFACT}."

    # 5) The third agent re-extracts it from ITS incoming message into ITS
    #    OWN registry — same URL, yet another independent local id.
    third_agent = _Ctx(text=args["request"], state={})
    user_links(third_agent)
    assert third_agent.state[USER_LINKS_STATE_KEY][link_id_for(ARTIFACT)]["url"] == ARTIFACT


def test_a_link_typed_without_a_scheme_survives_the_a2a_boundary():
    """A2A swaps the callee for a RemoteA2aAgent and nothing else: the seam is
    still the `request` argument of an AgentTool call, and the receiving side
    shares no session state at all.

    The scheme-less form is what makes this worth pinning. The user types
    `example.com`; what crosses the boundary is `https://example.com` — code
    wrote it, so it is schemed. Both must still be ONE link, and they are,
    because an id is the digest of the NORMALISED url: the remote derives the
    caller's id with nothing having travelled but the url itself.
    """
    caller = _Ctx(text="сложи результат в example.com/out", state={})
    user_links(caller)
    ref = link_ref(link_id_for("example.com/out"))
    assert ref in caller.state[LINKS_CONTEXT_STATE_KEY]

    args = {"request": f"выгрузи артефакт в {ref}"}
    resolve_link_refs(_Tool(), args, caller)
    assert args["request"] == "выгрузи артефакт в https://example.com/out"

    # The remote process: a fresh registry, nothing carried over.
    remote = _Ctx(text=args["request"], state={})
    user_links(remote)
    assert remote.state[USER_LINKS_STATE_KEY][link_id_for("example.com/out")]["url"] == (
        "https://example.com/out")
    # Same id on both sides, without a shared counter or shared state.
    assert set(remote.state[USER_LINKS_STATE_KEY]) == set(caller.state[USER_LINKS_STATE_KEY])

    # And the remote's own model reads the reference, not the url.
    req = _Request([args["request"]])
    redact_link_urls(callback_context=remote, llm_request=req)
    assert req.text() == f"выгрузи артефакт в {ref}"


# ── every agent captures, so no start_mode can switch the feature off ────────
def test_a_worker_builds_its_own_table_from_the_task_it_was_handed():
    """The caller's text reaches a worker with real URLs in it (code put them
    there), so the worker re-extracts them and can use references itself.

    This is what makes A2A work without shipping session state: the worker's
    registry is rebuilt from the message, not inherited from the caller.
    """
    worker = _Ctx(text=f"Обучи модель на {TRAIN} и проверь на {TEST}.", state={})
    user_links(worker)

    registry = worker.state[USER_LINKS_STATE_KEY]
    assert [e["url"] for e in registry.values()] == [TRAIN, TEST]
    assert REF_TRAIN in worker.state[LINKS_CONTEXT_STATE_KEY]
    assert TRAIN not in worker.state[LINKS_CONTEXT_STATE_KEY]

    # …and its own outbound calls can therefore use references.
    args = {"request": f"прогони {REF_TEST}"}
    resolve_link_refs(_Tool(), args, worker)
    assert args["request"] == f"прогони {TEST}"


def test_capture_does_not_depend_on_which_agent_is_root():
    """The original bug: capture was gated on `root: true`, and build_for_mode
    MOVES that flag, which silently emptied the registry in planner mode.

    There is no root special case left, so both start modes behave the same.
    """
    import copy
    from CoScientist.assembly import build_system, load_config

    def orchestrator_prompt_of(patch):
        config = copy.deepcopy(load_config())
        patch(config)
        agent = build_system(config=config).agents["OrchestratorAgent"]
        ctx = _Ctx(text=f"обучи на {TRAIN}", state={})
        chain = agent.before_agent_callback or []
        for callback in (chain if isinstance(chain, list) else [chain]):
            callback(ctx)
        return ctx.state.get(USER_LINKS_STATE_KEY) or {}

    def planner_mode(config):
        config.agents["PlanningPipelineAgent"].root = True
        config.agents["PlanningPipelineAgent"].enabled = True
        config.agents["OrchestratorAgent"].root = False

    def orchestrator_mode(config):
        config.agents["OrchestratorAgent"].root = True
        config.agents["PlanningPipelineAgent"].root = False
        config.agents["PlanningPipelineAgent"].enabled = False

    for patch in (planner_mode, orchestrator_mode):
        registry = orchestrator_prompt_of(patch)
        assert registry, f"{patch.__name__}: nothing captured"
        assert registry[ID_TRAIN]["url"] == TRAIN


def test_a_malformed_registry_cannot_take_the_run_down():
    """`user_links` runs on EVERY agent, so an unrenderable entry would kill the
    whole system rather than one call."""
    ctx = _Ctx(text="", state={USER_LINKS_STATE_KEY: {"linkbeef": "not-a-dict"}})
    user_links(ctx)

    assert ctx.state[LINKS_CONTEXT_STATE_KEY] == ""


def test_an_id_shaped_unlike_a_digest_still_renders():
    """Ids used to be parsed as ints to sort by, so a non-numeric one crashed
    the renderer. Nothing parses an id any more — an odd one just renders."""
    ctx = _Ctx(text="", state={USER_LINKS_STATE_KEY: {
        "train": {"id": "train", "url": "u", "role": "data file", "label": "t"}}})
    user_links(ctx)

    assert "[[train]]" in ctx.state[LINKS_CONTEXT_STATE_KEY]


# ── capture also scans the OTHER context blocks rendered for this turn ───────
def test_a_url_in_global_knowledge_memory_gets_a_reference():
    """graph_root's GLOBAL KNOWLEDGE MEMORY can quote a URL from a prior
    session's ingested facts. It was never in the incoming message, so
    scanning only user_content would leave the model with no [[linkN]] for a
    link it can plainly read — and it would retype the raw URL instead."""
    ctx = _Ctx(text="сравни результаты с прошлым запуском", state={
        "graph_root": f"GLOBAL KNOWLEDGE MEMORY — facts established in PRIOR runs:\n"
                      f"- dataset [Resource]: {TRAIN}",
    })
    user_links(ctx)

    registry = ctx.state[USER_LINKS_STATE_KEY]
    assert registry[ID_TRAIN]["url"] == TRAIN
    assert REF_TRAIN in ctx.state[LINKS_CONTEXT_STATE_KEY]


def test_the_attached_dataset_url_gets_a_reference_too():
    """inject_dataset_context renders the web-UI-attached dataset as a raw URL
    and tells the agent to pass it as `dataset_url` — without a reference, the
    model can only comply by copying that URL out by hand."""
    ctx = _Ctx(text="", state={
        "dataset_context": f"## Dataset attached to this session\n"
                           f"The user attached a dataset archive (.zip): {TRAIN}\n",
    })
    user_links(ctx)

    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN


def test_a_url_already_committed_as_evidence_is_reachable_via_research_context():
    ctx = _Ctx(text="", state={
        "research_context": f"Evidence E1: benchmark reported at {TEST}",
    })
    user_links(ctx)

    assert ctx.state[USER_LINKS_STATE_KEY][ID_TEST]["url"] == TEST


def test_message_and_context_links_land_in_one_registry_without_duplicates():
    ctx = _Ctx(text=f"продолжи работу с {TRAIN}", state={
        "graph_root": f"GLOBAL KNOWLEDGE MEMORY:\n- {TRAIN}",   # same link, restated
        "research_context": f"Evidence E1: {TEST}",              # a second, different link
    })
    user_links(ctx)

    registry = ctx.state[USER_LINKS_STATE_KEY]
    assert len(registry) == 2
    assert {e["url"] for e in registry.values()} == {TRAIN, TEST}


def test_an_agent_with_no_extra_context_blocks_behaves_as_before():
    # Most agents never populate graph_root/research_context/dataset_context;
    # capture must fall back cleanly to user_content alone.
    ctx = _Ctx(text=f"обучи на {TRAIN}", state={})
    user_links(ctx)

    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN


# ── after_tool: a link a TOOL returned gets registered too ───────────────────
def test_the_sandboxs_own_output_url_gets_a_reference():
    """This is the direction `user_links` cannot cover: a fresh artifact URL
    the sandbox invents mid-turn was never in the incoming message or in any
    context block rendered before the agent's turn began."""
    ctx = _Ctx(state={})
    result = {"status": "done", "artifact_url": TRAIN, "log": "training complete"}
    register_tool_result_links(_Tool(), {}, ctx, result)

    registry = ctx.state[USER_LINKS_STATE_KEY]
    assert registry[ID_TRAIN]["url"] == TRAIN
    # ...and the table is re-rendered in the SAME turn, so the agent's very
    # next model call already has a reference for it.
    assert REF_TRAIN in ctx.state[LINKS_CONTEXT_STATE_KEY]
    assert TRAIN not in ctx.state[LINKS_CONTEXT_STATE_KEY]


def test_a_plain_string_tool_result_is_scanned_too():
    ctx = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, ctx, f"Uploaded checkpoint to {TEST}")

    assert ctx.state[USER_LINKS_STATE_KEY][ID_TEST]["url"] == TEST


def test_tool_result_multiline_url_following_newline_is_registered():
    """A URL immediately preceded by a newline inside a markdown code block or
    multiline text field in a dict must be registered and not skipped by word boundary checks."""
    ctx = _Ctx(state={})
    summary = f"Results:\n```\n{TRAIN}\n```"
    register_tool_result_links(_Tool(), {}, ctx, {"status": "success", "summary": summary})

    assert ID_TRAIN in ctx.state[USER_LINKS_STATE_KEY]
    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN
    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["origin"] == "run_sandbox_task.summary"


def test_a_tool_result_is_described_by_where_it_came_back_not_by_its_json():
    """A mention is a slice of a SENTENCE, and a JSON body has none.

    Slicing one gave every result the same `…\\"follow_up_questions\\":null…`
    window — noise repeated per link in every later prompt of the turn. The
    key path says what that window was groping for, in a few characters.
    """
    ctx = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, ctx, {
        "query": "GSK-3 selectivity", "follow_up_questions": None,
        "results": [{"url": TRAIN, "title": "train"}, {"url": TEST, "title": "test"}]})

    registry = ctx.state[USER_LINKS_STATE_KEY]
    assert [e["mention"] for e in registry.values()] == ["", ""]
    assert registry[ID_TRAIN]["origin"] == "run_sandbox_task.results[0].url"
    assert registry[ID_TEST]["origin"] == "run_sandbox_task.results[1].url"

    block = ctx.state[LINKS_CONTEXT_STATE_KEY]
    assert "mentioned as" not in block and "follow_up_questions" not in block
    assert "returned as `run_sandbox_task.results[0].url`" in block


def test_the_sandboxs_two_links_are_told_apart_by_the_field_that_held_them():
    """`watch_url` and `vscode_url` are one host apart and nothing else.

    Both classify as `web page` on `10.32.2.2:8884`, so without the field name
    a model choosing "the live console" is guessing between two references.
    """
    watch = "http://10.32.2.2:8884/?task_id=7f3a91c2"
    vscode = "http://10.32.2.2:8884/vscode/7f3a91c2/?folder=/workspace"
    ctx = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, ctx, {
        "status": "ok", "sandbox_id": "7f3a91c2",
        # The summary repeats the console link in prose; the dedicated field
        # names it, and an exact match wins over one buried in a sentence.
        "summary": f"Готово. Следить можно тут: {watch}",
        "watch_url": watch, "vscode_url": vscode})

    registry = ctx.state[USER_LINKS_STATE_KEY]
    assert registry[link_id_for(watch)]["origin"] == "run_sandbox_task.watch_url"
    assert registry[link_id_for(vscode)]["origin"] == "run_sandbox_task.vscode_url"


def test_a_person_s_sentence_outranks_the_field_a_tool_returned_it_in():
    ctx = _Ctx(text=f"возьми обучающую выборку {TRAIN}", state={})
    register_tool_result_links(_Tool(), {}, ctx, {"artifact": TRAIN})
    user_links(ctx)

    entry = ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]
    assert entry["origin"] == "run_sandbox_task.artifact"
    assert "обучающую выборку" in entry["mention"]
    # Both are known; what the block shows is the human sentence — here as the
    # shared quote, since one link means one window.
    block = ctx.state[LINKS_CONTEXT_STATE_KEY]
    assert "обучающую выборку" in block and "returned as" not in block


def test_a_user_sentence_still_describes_a_link_a_tool_found_first():
    """Skipping the mention for a tool result must not cost the link the
    sentence a person later writes about it."""
    ctx = _Ctx(text=f"возьми обучающую выборку {TRAIN}", state={})
    register_tool_result_links(_Tool(), {}, ctx, {"url": TRAIN})
    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["mention"] == ""

    user_links(ctx)
    assert "обучающую выборку" in ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["mention"]


def test_a_tool_result_with_no_url_leaves_the_registry_untouched():
    ctx = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, ctx, {"status": "ok", "rows": 42})

    assert USER_LINKS_STATE_KEY not in ctx.state


def test_a_url_the_tool_already_registered_does_not_get_a_second_id():
    ctx = _Ctx(state=_registry_state())   # already holds both TRAIN and TEST
    register_tool_result_links(_Tool(), {}, ctx, {"artifact_url": TRAIN})

    assert len(ctx.state[USER_LINKS_STATE_KEY]) == 2
    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN


def test_the_freshly_registered_link_can_then_be_used_by_reference():
    """End to end: the sandbox returns a URL, the agent's next call writes a
    reference to it, and resolve_link_refs expands it back correctly."""
    ctx = _Ctx(state={})
    register_tool_result_links(
        _Tool(), {}, ctx, {"artifact_url": TRAIN, "status": "success"})

    args = {"request": f"приложи к отчёту {link_ref(link_id_for(TRAIN))}"}
    resolve_link_refs(_Tool(), args, ctx)
    assert args["request"] == f"приложи к отчёту {TRAIN}"


def test_a_malformed_tool_result_cannot_take_the_run_down():
    class _Unstringable:
        def __repr__(self):
            raise RuntimeError("boom")

    ctx = _Ctx(state={})
    register_tool_result_links(_Tool(), {}, ctx, _Unstringable())  # must not raise
    assert USER_LINKS_STATE_KEY not in ctx.state


# ── ids are digests of the url, not counter values ──────────────────────────
def test_two_agents_derive_the_same_id_for_the_same_link_independently():
    """No shared counter, no shared state, no coordination — and across an A2A
    process boundary session state does not travel at all. Agreement has to
    come from the url itself or not at all."""
    a = _Ctx(text=f"обучи на {TRAIN}", state={})
    user_links(a)
    # A different agent, different registry contents, met the link second.
    b = _Ctx(text=f"сначала {TEST}, потом {TRAIN}", state={})
    user_links(b)

    assert ID_TRAIN in a.state[USER_LINKS_STATE_KEY]
    assert ID_TRAIN in b.state[USER_LINKS_STATE_KEY]
    assert a.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == \
        b.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN


def test_position_in_the_message_does_not_change_an_id():
    """The counter it replaced made an id depend on arrival order, which is
    exactly what made two parallel branches collide on one number."""
    first = {}
    register_user_links(first, f"{TRAIN} и потом {TEST}")
    second = {}
    register_user_links(second, f"{TEST} и потом {TRAIN}")

    assert set(first[USER_LINKS_STATE_KEY]) == set(second[USER_LINKS_STATE_KEY])
    assert first[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN
    assert second[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN


def test_two_parallel_branches_discovering_different_links_cannot_collide():
    """The race this scheme exists to remove.

    ADK runs a turn's tool calls under asyncio.gather, each with an isolated
    state delta merged key-by-key afterwards. Under the old counter both
    branches computed the same next number from the same starting snapshot, so
    the merge kept one entry and silently dropped the other.
    """
    shared = {}
    register_user_links(shared, f"старт {TRAIN}")
    snapshot = dict(shared[USER_LINKS_STATE_KEY])

    # Two branches, each starting from the SAME snapshot, each finding its own
    # new link — the exact shape of the concurrent case.
    left = {USER_LINKS_STATE_KEY: dict(snapshot)}
    register_user_links(left, "первый воркер вернул https://minio/a.pt?sig=L")
    right = {USER_LINKS_STATE_KEY: dict(snapshot)}
    register_user_links(right, "второй воркер вернул https://minio/b.pt?sig=R")

    new_left = set(left[USER_LINKS_STATE_KEY]) - set(snapshot)
    new_right = set(right[USER_LINKS_STATE_KEY]) - set(snapshot)
    assert new_left and new_right
    assert not (new_left & new_right), "both branches claimed the same id"

    # Merging the two deltas the way ADK does keeps BOTH links.
    merged = {**left[USER_LINKS_STATE_KEY], **right[USER_LINKS_STATE_KEY]}
    assert len(merged) == 3
    assert {e["url"] for e in merged.values()} == {
        TRAIN, "https://minio/a.pt?sig=L", "https://minio/b.pt?sig=R"}


def test_a_digest_collision_is_resolved_by_lengthening_not_by_overwriting():
    """Correctness bound on the 1-in-65k case: two different urls must never
    share an id, even when their short digests match."""
    other = "https://example.org/other.zip"
    # Force the collision: seed the registry with a DIFFERENT url already
    # parked on the id TRAIN would want.
    taken = {ID_TRAIN: {"id": ID_TRAIN, "url": other,
                        "normalized": normalize_url(other)}}
    resolved = link_id_for(TRAIN, taken)

    assert resolved != ID_TRAIN
    assert resolved.startswith(ID_TRAIN)      # lengthened, same digest prefix
    # Deterministic: the same registry yields the same answer every time.
    assert link_id_for(TRAIN, taken) == resolved
    # A slot held by the SAME url is reused rather than lengthened — this is
    # what keeps re-registration idempotent.
    settled = {ID_TRAIN: {"id": ID_TRAIN, "url": TRAIN,
                          "normalized": normalize_url(TRAIN)}}
    assert link_id_for(TRAIN, settled) == ID_TRAIN


# ── before_model: the model must never SEE a url it could copy ──────────────
class _Request:
    """Stand-in for ADK's LlmRequest (only `.contents` is touched)."""

    def __init__(self, texts):
        self.contents = [
            type("C", (), {"parts": [_Part(t)]})() for t in texts
        ]

    def text(self):
        return "\n".join(p.text or "" for c in self.contents for p in c.parts)


def test_the_message_handed_to_the_model_carries_references_not_urls():
    """The leak the table redaction did not cover: the URL is in the MESSAGE.

    The user types one by hand, and — more often — we put one there ourselves,
    because resolve_link_refs expands the reference in the outbound `request`
    and that text becomes the callee's incoming message.
    """
    ctx = _Ctx(state=_registry_state())
    req = _Request([f"Обучи модель на {TRAIN}, проверь на {TEST}"])
    redact_link_urls(callback_context=ctx, llm_request=req)

    assert req.text() == f"Обучи модель на {REF_TRAIN}, проверь на {REF_TEST}"
    assert TRAIN not in req.text() and TEST not in req.text()


def test_redaction_leaves_the_registry_and_the_real_url_intact():
    """It rewrites the per-call copy only — the egress callbacks still have the
    real URL to substitute back in, which is what makes A2A work."""
    ctx = _Ctx(state=_registry_state())
    req = _Request([f"скачай {TRAIN}"])
    redact_link_urls(callback_context=ctx, llm_request=req)

    assert ctx.state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN
    args = {"request": req.text()}
    resolve_link_refs(_Tool(), args, ctx)
    assert args["request"] == f"скачай {TRAIN}"      # round-trips exactly


def test_an_unregistered_url_is_left_alone():
    # Only links we can substitute back are redacted; anything else must stay
    # readable or the model loses information it cannot recover.
    ctx = _Ctx(state=_registry_state())
    other = "https://unrelated.example.org/page"
    req = _Request([f"см. {other}"])
    redact_link_urls(callback_context=ctx, llm_request=req)

    assert req.text() == f"см. {other}"


def test_a_url_that_is_a_prefix_of_another_does_not_strand_its_tail():
    """Both a bare path and the same path with a signature can be registered;
    replacing the short one first would leave the long one's tail behind."""
    bare = TRAIN.split("?")[0]
    state = {}
    register_user_links(state, f"{bare} и {TRAIN}")
    ctx = _Ctx(state=state)
    req = _Request([f"полный {TRAIN} и короткий {bare}"])
    redact_link_urls(callback_context=ctx, llm_request=req)

    assert "http" not in req.text()
    assert req.text() == (f"полный {link_ref(link_id_for(TRAIN))} "
                          f"и короткий {link_ref(link_id_for(bare))}")


class _FnPart:
    """A function_response part: `.text` is None, the payload is a dict."""

    def __init__(self, response):
        self.text = None
        self.function_response = type("FR", (), {"response": response})()


class _ToolRequest:
    def __init__(self, part):
        self.contents = [type("C", (), {"parts": [part]})()]


def test_a_url_a_tool_returned_is_redacted_too():
    """The bigger half of the leak, and the one `part.text` never covered.

    A search tool answers with a JSON body of result urls. The table in the
    system prompt withheld those urls while the function response printed
    every one of them raw, two messages down — so the instruction not to
    retype a link was competing with the link itself, again.
    """
    body = {"results": [{"url": TRAIN, "title": "train"},
                        {"url": TEST, "title": "test"}],
            "answer": None, "images": []}
    state = {}
    register_tool_result_links(_Tool(), {}, _Ctx(state=state), body)

    part = _FnPart({"results": [{"url": TRAIN, "title": "train"},
                                {"url": TEST, "title": "test"}],
                    "answer": None, "images": []})
    redact_link_urls(callback_context=_Ctx(state=state),
                     llm_request=_ToolRequest(part))

    assert part.function_response.response["results"] == [
        {"url": REF_TRAIN, "title": "train"}, {"url": REF_TEST, "title": "test"}]
    # Untouched everywhere else: the registry still resolves the reference.
    assert state[USER_LINKS_STATE_KEY][ID_TRAIN]["url"] == TRAIN


def test_a_tool_result_with_nothing_registered_in_it_is_not_rebuilt():
    body = {"results": [{"url": "https://unrelated.example.org/page"}]}
    part = _FnPart(body)
    redact_link_urls(callback_context=_Ctx(state=_registry_state()),
                     llm_request=_ToolRequest(part))

    assert part.function_response.response is body


def test_redaction_is_a_noop_without_a_registry():
    ctx = _Ctx(state={})
    req = _Request([f"скачай {TRAIN}"])
    redact_link_urls(callback_context=ctx, llm_request=req)

    assert req.text() == f"скачай {TRAIN}"


def test_inject_original_query_before_redact_link_urls():
    """In planner mode, inject_original_query restores user_content, which then
    must be redacted by redact_link_urls."""
    from CoScientist.agents.callbacks import inject_original_query

    state = _registry_state()
    ctx = _Ctx(text=f"скачай {TRAIN} и запусти", state=state)
    req = _Request(["planner step-by-step roadmap"])

    # inject_original_query runs first
    inject_original_query(callback_context=ctx, llm_request=req)
    assert req.text() == f"скачай {TRAIN} и запусти"

    # redact_link_urls runs next
    redact_link_urls(callback_context=ctx, llm_request=req)
    assert req.text() == f"скачай {REF_TRAIN} и запусти"
    assert TRAIN not in req.text()


def test_planner_mode_callback_order():
    """Verify that in planner mode build_for_mode places inject_original_query before redact_link_urls."""
    from unittest.mock import patch
    from CoScientist.config import get_settings
    from CoScientist.agents import build_for_mode

    with patch.object(get_settings().web, "start_mode", "planner"):
        system = build_for_mode()
        orch = system.agents["OrchestratorAgent"]
        cb_list = orch.before_model_callback
        if not isinstance(cb_list, list):
            cb_list = [cb_list]
        cb_names = [getattr(cb, "__name__", str(cb)) for cb in cb_list]
        assert "inject_original_query" in cb_names
        assert "redact_link_urls" in cb_names
        inject_idx = cb_names.index("inject_original_query")
        redact_idx = cb_names.index("redact_link_urls")
        assert inject_idx < redact_idx, f"inject_original_query ({inject_idx}) must precede redact_link_urls ({redact_idx})"



