# 71762 후속 형식 진단 — 2026-11-13

현재 단계: 사전 고정한 검증셋 전용384건 진단을 완료했습니다.

이번에 한 일: 두 모델 Base/Tuned에 동일24검증문항(세 source 각8개), 기존/한국어 번역 지시문, 16/256토큰을 교차했습니다. 새 학습·test 추론·기존 점수 보정은 하지 않았습니다.

확인 결과: 유효응답은 기존 strict parser가 받아들인 형식입니다. 정확도 수치는 이 작은 검증 부분집합의 엄격한 문항 점수이며 임상 정확도나 새 test 성능이 아닙니다.

| 모델 | 조건 | 지시문 | token 상한 | 유효응답/24 | 엄격 정답/24 | 길이 종료 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| qwen3_17b | base | original | 16 | 22 | 8 | 0 |
| qwen3_17b | base | original | 256 | 22 | 8 | 0 |
| qwen3_17b | base | ko | 16 | 9 | 7 | 6 |
| qwen3_17b | base | ko | 256 | 9 | 7 | 0 |
| qwen3_17b | tuned | original | 16 | 0 | 0 | 24 |
| qwen3_17b | tuned | original | 256 | 0 | 0 | 11 |
| qwen3_17b | tuned | ko | 16 | 0 | 0 | 24 |
| qwen3_17b | tuned | ko | 256 | 0 | 0 | 11 |
| qwen3_4b | base | original | 16 | 21 | 13 | 2 |
| qwen3_4b | base | original | 256 | 21 | 13 | 0 |
| qwen3_4b | base | ko | 16 | 24 | 15 | 0 |
| qwen3_4b | base | ko | 256 | 24 | 15 | 0 |
| qwen3_4b | tuned | original | 16 | 0 | 0 | 24 |
| qwen3_4b | tuned | original | 256 | 0 | 0 | 7 |
| qwen3_4b | tuned | ko | 16 | 0 | 0 | 24 |
| qwen3_4b | tuned | ko | 256 | 0 | 0 | 6 |

실제 GPU 점유 0.272시간. 과거 파일208개 hash 보존을 확인했습니다. 같은 조건의 Base/Tuned native prompt가 일치했고, 16→256의 greedy token prefix는 192/192쌍 일치했습니다.

한계: 검증 문항24개뿐이며 기존에 이용 가능한 validation입니다. 번역은 native instruction token을 바꾸며, 문체·과제 분포·학습 노출량 차이와 무작위 QA 대응의 효과를 분리하지 못합니다. 기존 test 출력의 재분석은 관측 후 기술 통계이며 relaxed parser 정확도를 만들지 않았습니다.

다음 단계: 이 진단은 학습·정식 test 프로토콜을 자동 선택하지 않습니다. 결과에 따른 무한 재학습·문항 재추첨·추가seed를 실행하지 않습니다.

재개 명령(저장소에서): `PYTHONPATH=src ../.venv-medvlm/bin/python scripts/run_format_diagnostic_v5.py --data-root ../medvlm-data`

사용자가 할 일: 필수 입력은 없습니다. 전문가 검토는 미실행이며 Jetson은 사용자 보류입니다.
