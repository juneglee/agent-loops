from types import SimpleNamespace


def test_run_cells_factory_gives_each_repeat_its_own_seed(monkeypatch):
    import scripts.run_cells as rc

    made: list[dict] = []
    monkeypatch.setattr(rc, "LocalLLM", lambda tools, **kw: made.append(kw) or object())

    a = SimpleNamespace(base_url="u", model="m", temperature=0.7, seed=100)
    rc.make_factory(a, 0)([])
    rc.make_factory(a, 1)([])
    rc.make_factory(a, 2)([])

    assert [k["seed"] for k in made] == [100, 101, 102]
    assert all(k["temperature"] == 0.7 for k in made)


def test_run_single_measure_case_offsets_seed_per_repeat(monkeypatch):
    import scripts.run_single as rs

    made: list[dict] = []
    monkeypatch.setattr(
        rs, "LocalLLM", lambda tools, **kw: made.append(kw) or SimpleNamespace()
    )
    monkeypatch.setattr(rs, "single_tools", lambda case: [])

    rs.measure_case(
        {"id": "x"},
        None,
        {},
        base_url="u",
        model="m",
        category="simple",
        repeats=3,
        temperature=0.3,
        seed=5,
        _runner=lambda case, llm, cat, loop_module, loop_kwargs: SimpleNamespace(
            run_index=0
        ),
    )

    assert [k["seed"] for k in made] == [5, 6, 7]
    assert all(k["temperature"] == 0.3 for k in made)
