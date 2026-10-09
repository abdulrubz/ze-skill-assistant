"""The prompt store must serve Langfuse text when it can and the repo file
whenever it cannot -- and must never link a fallback to a Langfuse version."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import prompt_store
from app.skills.manifest import load_all_manifests


class _FakeClient:
    def __init__(self, *, text=None, version=3, is_fallback=False, raises=None):
        self.text, self.version, self.is_fallback, self.raises = text, version, is_fallback, raises
        self.calls = []

    def get_prompt(self, name, **kw):
        self.calls.append((name, kw))
        if self.raises:
            raise self.raises
        prompt = kw["fallback"] if self.is_fallback else self.text
        return SimpleNamespace(prompt=prompt, version=self.version, is_fallback=self.is_fallback)


@pytest.fixture
def use_client(monkeypatch):
    def _use(client):
        monkeypatch.setattr(prompt_store, "_client", lambda: client)
        return client
    return _use


def _repo(name):
    return prompt_store.read_file(prompt_store.catalog()[name])


def test_disabled_serves_repo_file_and_links_nothing(use_client):
    use_client(None)
    p = prompt_store.pipeline(prompt_store.PLANNER)
    assert p.text == _repo(prompt_store.PLANNER)
    assert p.handle is None
    assert isinstance(prompt_store.linked(p), type(prompt_store.linked()))


def test_langfuse_text_wins_and_is_linkable(use_client):
    client = use_client(_FakeClient(text="  edited in the UI  \n", version=7))
    p = prompt_store.pipeline(prompt_store.LEAD_IN)
    assert p.text == "edited in the UI"
    assert p.version == 7
    name, kw = client.calls[0]
    assert name == "pipeline/lead-in"
    assert kw["label"] == "production" and kw["type"] == "text"
    # the repo text rides along as the SDK-level fallback too
    assert kw["fallback"] == _repo(prompt_store.LEAD_IN)


def test_fetch_error_degrades_to_repo_file(use_client):
    use_client(_FakeClient(raises=ConnectionError("langfuse down")))
    p = prompt_store.pipeline(prompt_store.EXTRACTION)
    assert p.text == _repo(prompt_store.EXTRACTION)
    assert p.handle is None


def test_sdk_fallback_is_never_linked(use_client):
    use_client(_FakeClient(is_fallback=True))
    p = prompt_store.pipeline(prompt_store.LEAD_IN)
    assert p.text == _repo(prompt_store.LEAD_IN)
    assert p.handle is None


def test_blank_live_prompt_is_not_served(use_client):
    # An accidentally emptied prompt in the UI must not blank a system prompt.
    use_client(_FakeClient(text="   "))
    assert prompt_store.pipeline(prompt_store.PLANNER).text == _repo(prompt_store.PLANNER)


def test_braces_are_kept_verbatim(use_client):
    # compile() would treat {{x}} as a variable; the store must not.
    use_client(_FakeClient(text="Report {{amount}} as-is"))
    assert prompt_store.pipeline(prompt_store.PLANNER).text == "Report {{amount}} as-is"


def test_skill_playbook_reaches_the_rendered_system_prompt(use_client):
    from app.skills.factory import render_static_skill_prompt

    manifest = load_all_manifests()["order_analysis"]
    use_client(_FakeClient(text="EDITED PLAYBOOK"))
    rendered = render_static_skill_prompt(manifest)
    assert rendered.startswith("EDITED PLAYBOOK")
    # the manifest-derived rules are not editable and must survive any edit
    for caveat in manifest.caveats:
        assert caveat in rendered


def test_catalog_covers_pipeline_and_every_domain_skill_only():
    names = prompt_store.catalog()
    manifests = load_all_manifests().values()
    assert {prompt_store.PLANNER, prompt_store.EXTRACTION,
            prompt_store.LEAD_IN} <= names.keys()
    for m in manifests:
        key = prompt_store.skill_prompt_name(m.id)
        assert (key in names) == (m.kind == "domain"), m.id
    for path in names.values():
        assert path.exists() and prompt_store.read_file(path), path
