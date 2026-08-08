# agent-loops

논문에서 제안된 에이전트 루프를 동일한 인터페이스로 구현하고, 같은 모델과 평가 기준으로 비교한다.
각 논문의 원형 루프를 기준점으로 두고, 추가 요소를 적용했을 때 성능이 얼마나 달라지는지를 측정한다.

## 구성

스택 = 코어 루프 [+ 조합] [+ 하네스 층]. 예: `react`

### 코어 루프 (12)

각 루프는 독립된 파일로 구현하며, 논문 또는 저자 코드에서 정의한 핵심 동작을 테스트로 고정한다.

| 루프 | 구조 | 출처 | 참고 코드 |
|---|---|---|---|
| `single_call` | LLM 1회 → 반환된 tool call 실행 → 종료. 루프 없음 (하한선) | function calling | — |
| `react` | Thought → Action → Observation 반복, 관측에 따라 다음 행동 결정 | [2210.03629](https://arxiv.org/abs/2210.03629) | [ysymyth/ReAct](https://github.com/ysymyth/ReAct) |
| `plan_and_solve` | 전체 계획 생성 → 계획대로 단계 실행, 관측 기반 재계획 없음. 원 논문은 prompting 기법, 여기서는 도구 루프로 변형 | [2305.04091](https://arxiv.org/abs/2305.04091) | [AGI-Edgerunners/Plan-and-Solve-Prompting](https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting) |
| `rewoo` | 관측 피드백 없이 Plan 과 `#E` 의존성 생성 → Worker 실행 → Solver 가 evidence 종합 | [2305.18323](https://arxiv.org/abs/2305.18323) | [billxbf/ReWOO](https://github.com/billxbf/ReWOO) |
| `fixed_pipeline` | localization → repair → validation 고정 파이프라인(여기서는 locate → act → verify), 루프 없음 | [2407.01489](https://arxiv.org/abs/2407.01489) | [OpenAutoCoder/Agentless](https://github.com/OpenAutoCoder/Agentless) |
| `plan_and_execute` | 튜토리얼은 계획 → 한 step 실행 → 재계획. 여기서는 계획 전체 실행 후 재계획하는 변형 | LangGraph | [plan-and-execute tutorial](https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb) |
| `plan_and_act` | Planner 가 high-level plan 생성 → 별도 Executor 가 action 수행 → 관측 기반 재계획(동적 재계획 판) | [2503.09572](https://arxiv.org/abs/2503.09572) | [SqueezeAILab/plan-and-act](https://github.com/SqueezeAILab/plan-and-act) |
| `llm_compiler` | 함수 호출 DAG 생성 → 의존성이 풀린 task 실행(여기서는 순차) → joiner 가 종료/재계획 판정 | [2312.04511](https://arxiv.org/abs/2312.04511) | [SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler) |
| `adapt` | Executor 우선 실행 → 실패한 subtask 만 재귀 분해 (And/Or) | [2311.05772](https://arxiv.org/abs/2311.05772) | [archiki/ADaPT](https://github.com/archiki/ADaPT) |
| `codeact` | Python 코드 action → 실행 결과와 오류 관측 → 코드 수정 반복 | [2402.01030](https://arxiv.org/abs/2402.01030) | [xingyaoww/code-act](https://github.com/xingyaoww/code-act) |
| `reflexion` | trial → feedback → 언어적 reflection → memory → 다음 trial | [2303.11366](https://arxiv.org/abs/2303.11366) | [noahshinn/reflexion](https://github.com/noahshinn/reflexion) |
| `dfsdt` | branch 탐색 → 포기 시 sibling 분기 → 막히면 backtracking 하는 DFS | [2307.16789](https://arxiv.org/abs/2307.16789) | [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) |

## 실행

루프 하나를 llama-server 에 붙여 돌리거나, BFCL 칸 전체를 잰다.

```bash
pip install -e ".[dev,bench]"
python examples/run_loop.py --loop react --task "list the files in docs"
python scripts/run_cells.py --cells single_turn_single_step --loops react --limit 2
python scripts/run_tasks.py --tasks tests/fixtures/samples/tasks.json --loops react
python examples/demo.py
```

## 결과

공개 벤치마크의 공식 채점기로 잰 값. 성공률은 최종 파일 상태가 정답 상태와 같은 케이스의 비율이다.

| 칸 | 뜻 | 케이스 수 |
|---|---|---|
| 1T1S | 단일턴 싱글스텝. 요청 하나를 도구 호출 하나로 끝낸다 | 6 |
| 1TMS | 단일턴 멀티스텝. 요청 하나에 도구 호출 여러 개가 순서대로 필요하다 | 7 |
| MTMS | 멀티턴 멀티스텝. 요청이 여러 턴에 걸쳐 이어지고, 턴마다 여러 호출과 앞 턴의 상태가 필요하다 | 13 |

칸 값 = 성공률(%) / 케이스당 LLM 호출 수 / 케이스당 초

### 코어 루프

| 루프 | 1T1S | 1TMS | MTMS |
|---|---|---|---|
| `single_call` | 50.0 / 1.0 / 12 | 0.0 / 1.0 / 12 | 0.0 / 3.4 / 35 |
| `react` | 83.3 / 2.7 / 9 | 57.1 / 6.3 / 34 | 23.1 / 19.9 / 99 |
| `plan_and_solve` | 66.7 / 1.0 / 9 | 14.3 / 1.0 / 9 | 0.0 / 3.4 / 26 |
| `rewoo` | 66.7 / 2.0 / 19 | 28.6 / 2.0 / 15 | 0.0 / 6.8 / 66 |
| `fixed_pipeline` | 83.3 / 3.0 / 28 | 28.6 / 3.0 / 21 | 7.7 / 10.2 / 98 |
| `plan_and_execute` | 16.7 / 1.5 / 6 | 14.3 / 2.0 / 10 | 7.7 / 7.1 / 45 |
| `plan_and_act` | 33.3 / 3.0 / 14 | 0.0 / 3.0 / 18 | 0.0 / 11.5 / 85 |
| `llm_compiler` | 83.3 / 2.5 / 10 | 42.9 / 2.7 / 16 | 7.7 / 8.8 / 57 |
| `adapt` | 83.3 / 8.0 / 21 | 71.4 / 10.6 / 71 | 23.1 / 42.5 / 211 |
| `codeact` | 50.0 / 2.7 / 24 | 28.6 / 2.0 / 15 | 0.0 / 11.4 / 136 |
| `reflexion` | 83.3 / 9.8 / 76 | 71.4 / 8.0 / 64 | 38.5 / 25.6 / 214 |
| `dfsdt` | 83.3 / 4.3 / 15 | 85.7 / 5.1 / 23 | 38.5 / 14.8 / 98 |

## 환경

모든 루프를 같은 모델, 같은 서버 설정, 같은 데이터로 돌린다.

| 항목 | 설정 |
|---|---|
| 모델 | `gemma-4-E4B-it QAT q4_0 GGUF` — https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf |
| 추론 런타임 | `llama.cpp llama-server` (OpenAI 호환 API) — https://github.com/ggml-org/llama.cpp |
| 런타임 설정 | `-c 32768 -ngl 99 -fa on --cache-reuse 256 --jinja` |
| 내부 평가셋 | 실파일 워크스페이스 과제 (`tests/fixtures/samples`), 최종 파일 상태 기준 평가 |
| 외부 벤치마크 | BFCL v4 `multi_turn_base` 파일 관리 과제, `bfcl-eval` 공식 평가 방식 사용 — https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard |
| 하드웨어 | MacBook Pro, Apple M2 Pro, 32 GB, Metal |

## 라이선스

Apache-2.0
