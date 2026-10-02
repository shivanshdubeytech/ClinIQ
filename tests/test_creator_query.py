"""Tests for anchored creator and identity queries in src/pipeline.py."""
import pytest
from src.pipeline import check_creator_query


def test_pacemaker_is_not_creator_query():
    assert not check_creator_query("What is a pacemaker?")
    assert not check_creator_query("How does a pacemaker work?")


def test_dad_symptom_is_not_creator_query():
    assert not check_creator_query("My dad is feeling dizzy and nauseous.")
    assert not check_creator_query("What causes father's knee pain?")


def test_grandfather_is_not_creator_query():
    assert not check_creator_query("My grandfather has diabetes and memory loss.")


def test_who_is_your_father_is_creator_query():
    assert check_creator_query("Who is your father?")
    assert check_creator_query("who is your creator")
    assert check_creator_query("Who's your developer?")


def test_who_created_you_is_creator_query():
    assert check_creator_query("Who created you?")
    assert check_creator_query("who made you")
    assert check_creator_query("who developed you")
