# 측정 기록

측정 하나가 문서 하나다. 파일명은 `Result_<벤치>.md` 이고, 다시 재면 같은 문서에 날짜를 단 절을 더해 이전 값을 남긴다.
각 문서는 머리에 조건(모델·서버 옵션·도구 세트·스텝 예산·채점 방식·코드 커밋)을 고정해 적는다.
과제를 고르고 형식을 옮기고 하네스를 고친 과정은 [Process.md](Process.md) 에 있다.

| 문서 | 과제 | 대상 | 상태 |
|---|---|---|---|
| [Result_BFCL](Result_BFCL.md) | BFCL v4 multi_turn_base 파일 관리 26 | 코어 12 · 조합 9 · 하네스 2 | 완료 |
| [Result_MCPMark](Result_MCPMark.md) | MCPMark Filesystem 파일 관리 21 | 코어 12 | 1차 완료, 잠정 |
| [Result_InterCode](Result_InterCode.md) | InterCode-Bash 파일시스템 20 | 코어 12 | 1차 완료, 잠정 |
| [Analysis_FileManagement](Analysis_FileManagement.md) | 위 두 세트 합산 실패 분석 279건 | 코어 12 | 완료 |

## 읽는 법

- 성공은 채점기가 통과로 판정한 케이스 수다. 과제마다 채점기가 다르므로 문서 안에서만 비교한다.
- 호출 수는 케이스당 LLM 호출, 툴 실행은 케이스당 실제 실행된 도구 호출이다.
- 시간은 케이스당 초이고 thinking 생성을 포함한다. 속도는 llama-server 의 `timings.predicted_per_second` 평균(TPS)으로 적는다.
- 케이스 수가 적은 측정은 1~2건 차이가 잡음 범위다. 문서마다 유보 조항을 둔다.
