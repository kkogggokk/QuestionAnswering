# 1. 프로젝트 개요
| 항목 | 내용 |
| --- | --- |
| 기간 | 2021.11.02 – 2021.11.15 (2주) |
| 팀원 | 안희진 김소연 김지수 정영빈 |
| 주제 | 한국어 지문 질의응답 모델 설계 및 성능 최적화 |
| 내용 | 본 프로젝트는 한국어 지문을 기반으로 질문에 적합한 답변을 생성하는 MRC 모델 개발을 목표로 진행되었습니다. EDA를 통해 도메인 불균형 문제를 확인하고, 데이터 증강(AIHub 데이터 활용)과 토큰 길이 기반 전처리를 수행했습니다. BERT와 RoBERTa 모델을 학습하여 성능을 비교하며, 하이퍼파라미터 튜닝과 Hard Voting 앙상블 기법으로 최적화했습니다. 최종적으로 편집거리 2.59의 성능을 달성하며 MRC 모델의 성능을 향상시켰습니다. |
| 결과 | - 편집거리 2.59 <br>- 교육과정 캐글 대회에서 8팀 중 1위 차지 |
| 기술스택 | 프레임워크 및 라이브러리: Pytorch, Huggingface Transformers, WandB <br>데이터 처리 및 분석: Numpy, Pandas, Matplotlib <br>모델 및 기법: <br>- Transformer 기반 모델: BERT (klue/bert-base), RoBERTa (klue/roberta-base) <br>- 기법: 데이터 증강 (AIHub 데이터 활용), 하이퍼파라미터 튜닝 (Optimizer: RAdam, AdamW; Scheduler 사용), 후처리 (답변 길이 제한), 앙상블(Hard Voting) <br>협업 및 기타도구: Github, Kaggle, Jupyter Notebook |
| 코드URL | [https://github.com/sykverse/Question-Answering](https://github.com/sykverse/Question-Answering)<br>[https://www.kaggle.com/competitions/goormkoreanmrcproject](https://www.kaggle.com/competitions/goormkoreanmrcproject)     |

# 2. 프로젝트 진행 프로세스

## 2.1 EDA 및 Preprocessing

![EDA 및 Preprocessing](https://raw.githubusercontent.com/kkogggokk/QuestionAnswering/refs/heads/main/images/Screenshot_2024-11-25_at_12.59.43_PM.png)

- **동일 Context와 Question에 대한 다른 Answers 문제**<br>동일 지문과 질문에 대해 여러 답변이 존재하는 약 1만 개의 데이터 확인. 평가 지표가 답변 길이에 민감하므로, 가장 짧은 답변만을 선택한 방식과 모든 답변을 활용한 방식을 비교. 결과적으로 모든 데이터를 활용한 방법이 더 나은 성능을 기록.
- **Data Augmentation**<br>AIHub MRC 데이터를 활용하여 짧은 문장, 긴 문장, 두 가지 형태 모두를 추가하는 방식으로 실험. 짧은 문장만을 추가했을 때 가장 높은 성능을 달성했으나, 도메인 불일치로 인해 증강 후 성능 저하 발생. 
- **Split Ratio**<br>Train과 Dev 데이터를 8:2로 분할한 경우와 모든 데이터를 Train으로 사용하는 경우 비교. 모든 데이터를 Train에 활용한 경우 더 나은 성능을 기록.

## 2.2 모델 선정 및 분석

![모델 선정 및 분석](https://raw.githubusercontent.com/kkogggokk/QuestionAnswering/refs/heads/main/images/Screenshot_2024-11-25_at_12.58.58_PM.png)

- Pretrained 모델로 KLUE/BERT-base와 KLUE/RoBERTa-base를 선택.
    - 주어진 데이터만 학습 시 BERT가 더 높은 성능을 기록
    - 데이터 증강 후에는 RoBERTa가 더 많은 파라미터로 인해 더 좋은 성능을 보임.
- 모델 한계:Extraction-based 모델 외에 KoELECTRA와 T5와 같은 다른 모델을 실험하지 못한 점은 한계로 남음.

## 2.3 모델 평가 및 개선

![모델 평가 및 개선](https://raw.githubusercontent.com/kkogggokk/QuestionAnswering/refs/heads/main/images/Screenshot_2024-11-25_at_12.58.27_PM.png)

- Hyper-parameter Tuning:Batch Size, Epoch, Optimizer, Learning Rate, Scheduler 등을 조합해 실험. RAdam과 Scheduler를 추가했을 때 더 좋은 성능을 보임.
- Evaluation Metrics:편집거리와 EM Score를 지표로 활용.
- 답변 길이 자르기:답변 길이를 앞, 뒤에서 제한하는 후처리 기법으로 최적화를 진행. 뒤에서 12글자를 자른 경우 가장 높은 성능 기록.
- Ensemble(Hard Voting):약 10개의 모델을 학습시키고 Hard Voting 방식으로 최적의 답변을 결정, 성능을 추가적으로 향상시킴.

# 3. 프로젝트 결과

- 편집거리 2.59334 기록
- 교육과정 캐글 대회에서 8팀 중 1위 차지

# 4. 자체 평가 및 보완

### Pre-processing

- 한계: 512 토큰 이상 긴 문장을 처리하는 전처리가 부족.
- 개선방향: 긴 문장을 효과적으로 처리할 수 있는 전처리 기법 개발, 모델의 max_token_length에 맞게 효과적인 input을 만들어주는 방법 고려

### Data Augmentation

- 한계: 도메인 분포를 고려하지 않은 Random Sampling 기반의 데이터 증강.
- 개선방향: 도메인 분포를 고려한 데이터 증강 기법 연구

### Model

- 한계: Extraction-based 모델만 실험해 다른 유형의 모델을 다루지 못함.
- 개선방향: KoELECTRA, T5 등 다양한 모델 활용.

### Others

- 한계: Train-Validation Split을 한 조합만 실험한 점.
- 개선방향: K-Fold Validation 도입.
