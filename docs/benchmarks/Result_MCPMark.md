# MCPMark Filesystem 파일 관리 21과제 — 코어 루프 12

## 조건

| 항목 | 값 |
|---|---|
| 과제 | MCPMark Filesystem 40 중 파일 관리 21 (`data/tasks/mcpmark_fs`, 원 저작 eval-sys/mcpmark, Apache-2.0). 문서 해석·계산 위주 19건은 제외 |
| 채점 | 과제별 `verify.py` (원본 그대로) |
| 모델 | gemma-4-E4B-it QAT q4_0 GGUF |
| 서버 | llama.cpp llama-server, `-c 32768 -ngl 99 -fa on --cache-reuse 256 --jinja --reasoning-budget 2048` (thinking 켬, 하네스는 reasoning 내용을 버림) |
| 도구 | `fs_tool` 6종 (Read·Write·Edit·Bash·Glob·Grep), 관측은 평문 |
| 예산 | `--max-steps 30 --max-tokens 8192`, 셸 명령 10초 |
| 가드레일 | 셸은 macOS `sandbox-exec` 로 워크스페이스 밖 쓰기 차단 |
| 실행 | `python scripts/fetch_mcpmark_fixtures.py` 로 워크스페이스를 받은 뒤 `python scripts/run_tasks.py --tasks data/tasks/mcpmark_fs --select <21건 id> --tools fs_tool --loops <루프> --max-steps 30 --keep-workspaces` |
| 반복 | 1회 |
| 코드 | d30d1f6 |

가드가 명령을 거부하던 케이스는 샌드박스 조건으로 다시 재어 반영했다(react +1). 호출 수·시간·종료 사유는 재측정 전 실행의 값이다.

## 결과

| 루프 | 성공 / 21 | LLM 호출/케이스 | 툴 실행/케이스 | 초/케이스 | TPS | 종료 사유 |
|---|---|---|---|---|---|---|
| `react` | 9 | 11.5 | 10.6 | 134 | 50 | success 17, max_steps 2, no_action 2 |
| `codeact` | 7 | 11.1 | 122.4 | 3130 | 39 | success 14, no_action 3, max_steps 2, truncated 2 |
| `dfsdt` | 6 | 18.8 | 18.0 | 258 | 54 | max_steps 10, success 9, give_up 1, truncated 1 |
| `plan_and_execute` | 5 | 13.3 | 12.9 | 198 | 50 | success 13, max_steps 6, no_plan 2 |
| `reflexion` | 5 | 13.1 | 11.4 | 131 | 55 | success 16, max_trials 3, no_action 2 |
| `plan_and_solve` | 4 | 1.0 | 2.5 | 28 | 49 | success 21 |
| `adapt` | 3 | 14.7 | 9.9 | 186 | 58 | success 10, max_steps 4, no_action 3, parse_fail 3, max_depth 1 |
| `plan_and_act` | 3 | 15.6 | 7.0 | 268 | 48 | parse_fail 11, success 10 |
| `llm_compiler` | 2 | 3.4 | 4.3 | 61 | 49 | max_steps 12, success 5, parse_fail 4 |
| `rewoo` | 2 | 2.0 | 5.1 | 46 | 49 | success 21 |
| `single_call` | 1 | 1.0 | 1.0 | 16 | 53 | success 21 |
| `fixed_pipeline` | 1 | 3.0 | 3.0 | 35 | 49 | success 21 |

종료 사유의 `success` 는 루프가 스스로 완료를 선언했다는 뜻이지 채점 통과가 아니다. `single_call`·`rewoo`·`fixed_pipeline`·`plan_and_solve` 는 21건 전부 완료를 선언했지만 통과는 1~4건이다.

## 통과 과제

- `react`: file_context/pattern_matching, file_context/uppercase, file_property/largest_rename, folder_structure/structure_analysis, legal_document/file_reorganize, papers/papers_counting, student_database/recommender_name, standard file_context/file_splitting, desktop_template/time_classification
- `codeact`: file_context/file_splitting, file_context/pattern_matching, file_context/uppercase, file_property/txt_merging, folder_structure/structure_analysis, student_database/recommender_name, standard file_context/file_splitting
- `dfsdt`: file_property/largest_rename, folder_structure/structure_analysis, legal_document/file_reorganize, papers/papers_counting, student_database/duplicate_name, student_database/recommender_name
- `plan_and_execute`: file_context/uppercase, file_property/largest_rename, folder_structure/structure_analysis, legal_document/file_reorganize, standard file_context/file_splitting
- `reflexion`: file_context/uppercase, folder_structure/structure_analysis, legal_document/file_reorganize, student_database/recommender_name, standard file_context/file_splitting
- `plan_and_solve`: file_context/file_splitting, file_context/uppercase, legal_document/file_reorganize, papers/papers_counting
- `adapt`: file_context/file_splitting, folder_structure/structure_analysis, student_database/recommender_name
- `plan_and_act`: folder_structure/structure_analysis, legal_document/file_reorganize, student_database/duplicate_name
- `llm_compiler`: file_context/uppercase, legal_document/file_reorganize
- `rewoo`: file_context/uppercase, folder_structure/structure_analysis
- `single_call`: file_context/file_merging
- `fixed_pipeline`: papers/papers_counting

## 읽는 법

- 여러 파일을 순서대로 다뤄야 하는 과제라 관측을 보고 다음 행동을 정하는 루프(`react`·`codeact`·`dfsdt`·`plan_and_execute`·`reflexion`)만 5건 이상 통과했다. 계획을 한 번에 고정하는 루프는 4건 이하다.
- `codeact` 의 시간은 스텝을 다 쓰며 코드 실행을 반복한 소수 케이스가 만든 값이다. 밤사이 머신 수면이 섞인 케이스가 있어 `codeact`·`plan_and_act` 의 시간은 상한으로만 읽는다.
- `plan_and_act` 는 Executor 가 툴 호출을 텍스트로 내는 형식 오류(parse_fail 11)로 절반이 조기 종료됐다.

## 유보

21과제 1회 실행이라 1~2건 차이는 잡음이다.
