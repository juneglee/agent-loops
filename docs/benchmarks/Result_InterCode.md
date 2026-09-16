# InterCode-Bash 파일시스템 20과제 — 코어 루프 12

## 조건

| 항목 | 값 |
|---|---|
| 과제 | InterCode-Bash 파일시스템 과제 26 중 20 (`data/tasks/intercode_bash`, 원 저작 princeton-nlp/intercode, MIT). 정답 명령이 요청 문장과 모순되는 6건 제거 |
| 채점 | 정답 셸 명령을 재생한 최종 상태와 비교(파일 집합·내용, tar 는 내용 다이제스트, 심링크는 대상 경로) |
| 모델 | gemma-4-E4B-it QAT q4_0 GGUF |
| 서버 | llama.cpp llama-server, `-c 32768 -ngl 99 -fa on --cache-reuse 256 --jinja --reasoning-budget 2048` (thinking 켬) |
| 도구 | `fs_tool` 6종, 관측은 평문 |
| 예산 | `--max-steps 30 --max-tokens 8192`, 셸 명령 10초 |
| 가드레일 | 셸은 macOS `sandbox-exec` 로 워크스페이스 밖 쓰기 차단 |
| 실행 | `python scripts/run_tasks.py --tasks data/tasks/intercode_bash --tools fs_tool --loops <루프> --max-steps 30 --keep-workspaces` |
| 반복 | 1회 |
| 코드 | d30d1f6 |

가드가 명령을 거부하던 케이스는 샌드박스 조건으로 다시 재어 반영했다(react +1, single_call +1, dfsdt −1, fixed_pipeline −1). 호출 수·시간·종료 사유는 재측정 전 실행의 값이다.

## 결과

| 루프 | 성공 / 20 | LLM 호출/케이스 | 툴 실행/케이스 | 초/케이스 | TPS | 종료 사유 |
|---|---|---|---|---|---|---|
| `react` | 17 | 3.0 | 1.6 | 35 | 51 | success 19, no_action 1 |
| `plan_and_execute` | 15 | 2.6 | 1.7 | 15 | 51 | success 20 |
| `reflexion` | 15 | 3.6 | 1.5 | 52 | 51 | success 18, no_action 2 |
| `plan_and_solve` | 15 | 1.0 | 1.3 | 8 | 49 | success 20 |
| `codeact` | 14 | 5.2 | 4.2 | 71 | 47 | success 20 |
| `adapt` | 14 | 4.5 | 2.5 | 56 | 60 | success 17, no_action 2, max_steps 1 |
| `llm_compiler` | 14 | 2.7 | 1.6 | 20 | 50 | success 15, max_steps 5 |
| `dfsdt` | 13 | 4.0 | 2.9 | 29 | 57 | success 18, give_up 1, max_steps 1 |
| `rewoo` | 13 | 2.0 | 1.4 | 15 | 50 | success 20 |
| `single_call` | 13 | 1.0 | 1.0 | 6 | 51 | success 20 |
| `plan_and_act` | 11 | 4.4 | 1.2 | 24 | 51 | success 13, parse_fail 7 |
| `fixed_pipeline` | 11 | 3.0 | 2.9 | 23 | 49 | success 20 |

## 읽는 법

- 명령 한 번으로 끝나는 과제라 루프 간 차이가 작다(11~17). `single_call` 도 13건을 통과한다.
- 차이는 첫 명령이 틀렸을 때 관측을 보고 고치는지에서 난다. 과제 문장의 "test directory" 를 `test/` 폴더로 읽는 오해가 대표적이고, `react` 는 `ls -F` 로 실제 위치를 확인해 회복했지만 `plan_and_solve`·`rewoo` 는 같은 명령을 반복하다 끝났다.
- 통과에 필요한 툴 실행이 1~2회라 계획·분해·반성에 쓰는 호출은 그대로 비용이다. `plan_and_act` 는 형식 오류(parse_fail 7)로 가장 낮다.

## 유보

20과제 1회 실행이라 1~2건 차이는 잡음이다.
