# agent-loops

논문에서 제안된 에이전트 루프를 동일한 인터페이스로 구현하고, 같은 모델과 평가 기준으로 비교한다.
각 논문의 원형 루프를 기준점으로 두고, 추가 요소를 적용했을 때 성능이 얼마나 달라지는지를 측정한다.

## 환경

모든 루프를 같은 모델, 같은 서버 설정, 같은 데이터로 돌린다.

| 항목 | 설정 |
|---|---|
| 모델 | `gemma-4-E4B-it QAT q4_0 GGUF` — https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf |
| 추론 런타임 | `llama.cpp llama-server` (OpenAI 호환 API) — https://github.com/ggml-org/llama.cpp |
| 런타임 설정 | `-c 32768 -ngl 99 -fa on --cache-reuse 256 --jinja` |
| 외부 벤치마크 | BFCL v4 `multi_turn_base` 파일 관리 과제, `bfcl-eval` 공식 평가 방식 사용 — https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard |
| 하드웨어 | MacBook Pro, Apple M2 Pro, 32 GB, Metal |

## 라이선스

Apache-2.0
