"""Verify label isolation and strict interpretation of shuffled LLM rankings offline."""

import json

import pytest

from asr_rescoring.llm_scoring import api_request, make_request, parse_ranking, presentation_order


def test_prompt_contains_candidates_but_no_gold_label():
    settings = {"model": "local", "think": False, "keep_alive": "5m", "options": {}}
    payload, order = make_request(settings, "sample", ("Перший текст.", "Інший текст!"), 42)
    candidates = json.loads(payload["messages"][1]["content"])
    assert {row["text"] for row in candidates} == {"перший текст", "інший текст"}
    assert '"reference":' not in json.dumps(payload)
    assert all(set(row) == {"number", "text"} for row in candidates)
    assert order == presentation_order("sample", 2, 42)


def test_displayed_ranking_maps_back_to_original_indices():
    response = {"done": True, "done_reason": "stop", "message": {"content": '{"ranking":[2,1,3]}'}}
    assert parse_ranking(response, [2, 0, 1]) == [0, 2, 1]


@pytest.mark.parametrize("ranking", [[1, 1, 3], [1, 2], [0, 1, 2], [1, 2, 4], [True, 2, 3]])
def test_invalid_ranking_is_never_replaced_with_first_candidate(ranking):
    response = {"done": True, "message": {"content": json.dumps({"ranking": ranking})}}
    with pytest.raises(ValueError, match="permutation"):
        parse_ranking(response, [2, 0, 1])


def test_truncated_output_is_rejected():
    with pytest.raises(ValueError, match="truncated"):
        parse_ranking({"done": True, "done_reason": "length"}, [0])


def test_remote_endpoint_is_rejected_without_network_access():
    with pytest.raises(ValueError, match="local"):
        api_request("https://example.com", "/api/chat", {})
