from agent_loops.loops.plan_and_solve import parse_plan


def test_plan_and_solve_shares_the_same_typing():
    plan = parse_plan("1. calc[base=10, height=5]")

    assert plan == [("calc", {"base": 10, "height": 5})]


def test_plan_and_solve_strips_quotes():
    plan = parse_plan('1. f[s="hi", n=3]')

    assert plan[0][1] == {"s": "hi", "n": 3}
