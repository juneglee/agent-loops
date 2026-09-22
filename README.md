# agent-loops

논문에서 제안된 에이전트 루프를 동일한 인터페이스로 구현하고, 같은 모델과 평가 기준으로 비교한다.
각 논문의 원형 루프를 기준점으로 두고, 추가 요소를 적용했을 때 성능이 얼마나 달라지는지를 측정한다.

## 구성

스택 = 코어 루프 [+ 오케스트레이션 조합] [+ 하네스 기능]. 예: `react`, `planner+react`, `react+todo`

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

### 오케스트레이션 조합 (11)

코어 루프를 worker 로 두고 그 실행을 상위에서 제어하는 구조다. worker 구현은 수정 없이 그대로 사용한다.

| 오케스트레이션 | 구조 | 파일 |
|---|---|---|
| `planner+<worker>` | 계획기가 과제를 하위 과제로 나누고, 각 하위 과제를 worker 루프에 맡긴 뒤 결과를 보고 재계획 | `compose/hierarchical.py` |
| `adaptive+<worker>` | worker 를 먼저 그대로 돌리고, 실패 선언이나 도구 오류가 나면 계획기가 개입해 분해 | `compose/adaptive.py` |
| `routed+<worker>` | 게이트 1콜이 과제를 단순/복잡으로 판정해 단순이면 worker 직행, 복잡이면 planner 경로 | `compose/routed.py` |

worker: `react`, `single_call`, `codeact`, `dfsdt` (`dfsdt` 는 `planner`, `routed` 만)

### 하네스 기능 (2)

하네스는 모델을 실행하는 데 필요한 주변 구조다. 프롬프트 구성, 도구와 도구 설명, 모델 호출, 실행 환경 연결, 그리고 실행 환경과 모델 호출을 감싸는 층이 여기에 들어간다. 보통은 에이전트 루프도 하네스에 넣지만, 이 저장소는 루프를 비교하는 것이 목적이라 루프는 코어 루프로 따로 뗐다. 아래 두 기능은 켜고 끌 수 있는 층으로, 작업 상태 관리와 실행 결과 검증을 더한다. 둘 다 별도의 LLM 호출 없이 실행 환경과 모델 호출을 바깥에서 감싸기 때문에 어떤 루프와 조합에도 똑같이 적용된다.

| 기능 | 동작 | 파일 |
|---|---|---|
| `+todo` | `update_todo` 도구로 관리하는 할 일 목록을 매 호출 프롬프트에 함께 전달한다 | `harness/todo.py` |
| `+verifier` | 도구 호출 결과를 사후조건으로 검사해, 조용히 실패한 호출을 오류 관측으로 바꿔 루프에 알린다 | `harness/verifier.py` |

## 실행

```bash
pip install -e ".[dev,bench]"
python examples/run_loop.py --loop react --task "list the files in docs"
python examples/demo.py
```

측정별 실행 명령은 각 측정 문서의 조건 표에 있다.

## 결과

수치는 측정마다 [docs/benchmarks](docs/benchmarks/README.md) 에 문서 하나로 둔다. 여기서는 무엇을 확인하려는지와 진행 상태만 적는다.

**확인하려는 것**

- 같은 4B 온디바이스 모델, 같은 도구 6종, 같은 스텝 예산에서 12개 루프가 실제 파일 관리 요청을 얼마나 해내는가.
- 논문에서 좋은 루프(계획·반성·탐색)가 짧고 구체적인 파일 과제에서도 이득인지, 아니면 호출만 늘리는지.
- 4B 모델이 어디서 깨지는지. 형식·채널 이탈, 안 하고 완료 선언, 예산 소진 같은 실패 패턴이 루프마다 어떻게 다른지.

**어떻게 재는가**

- 과제는 공개 데이터(MCPMark Filesystem, InterCode-Bash)를 고치지 않고 쓰고, 채점은 공식 검증 스크립트와 최종 파일 상태 비교로 한다.
- 루프 로직은 논문·저자 코드 그대로 두고, 하네스(프롬프트·도구·층)만 공통으로 맞춘다.
- 가드레일: 셸 쓰기를 OS 샌드박스로 워크스페이스 안에 가둔다.

**진행**

| 측정 | 상태 | 문서 |
|---|---|---|
| BFCL multi_turn 파일 관리 26과제, 12루프·조합·하네스 | 완료 | [문서](docs/benchmarks/Result_BFCL.md) |
| MCPMark Filesystem 파일 관리 21과제, 12루프 | 1차 완료, 잠정 | [문서](docs/benchmarks/Result_MCPMark.md) |
| InterCode-Bash 파일시스템 20과제, 12루프 | 1차 완료, 잠정 | [문서](docs/benchmarks/Result_InterCode.md) |
| 위 41과제 실패 분석 279건 | 완료 | [문서](docs/benchmarks/Analysis_FileManagement.md) |
| 한국어 파일 관리 요청 300건(단일 250, 멀티턴 50), 상위 7루프 | 완료, 2,100건 전건 판정 | [문서](docs/benchmarks/Result_KOFM.md) |

## 환경

모든 루프를 같은 모델, 같은 서버 설정, 같은 데이터로 돌린다.

| 항목 | 설정 |
|---|---|
| 모델 | `gemma-4-E4B-it QAT q4_0 GGUF` — https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf |
| 추론 런타임 | `llama.cpp llama-server` (OpenAI 호환 API) — https://github.com/ggml-org/llama.cpp |
| 런타임 설정 | `-c 32768 -ngl 99 -fa on --cache-reuse 256 --jinja` |
| 내부 평가셋 | 실파일 워크스페이스 과제 (`tests/fixtures/samples`), 최종 파일 상태 기준 평가 |
| 공개 파일관리 과제 | MCPMark Filesystem 40건 (`data/tasks/mcpmark_fs`, 원 저작 eval-sys/mcpmark, Apache-2.0, L1 10 / L3 30), 워크스페이스 10종은 `scripts/fetch_mcpmark_fixtures.py` 로 내려받고 과제별 `verify.py` 로 평가 — https://github.com/eval-sys/mcpmark |
| 외부 벤치마크 | BFCL v4 `multi_turn_base` 파일 관리 과제, `bfcl-eval` 공식 평가 방식 사용 — https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard |
| 하드웨어 | MacBook Pro, Apple M2 Pro, 32 GB, Metal |

## 라이선스

Apache-2.0
