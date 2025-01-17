- [알고 있는 metric에 대해 설명해주세요. (ex. RMSE, MAE, recall, precision ...)](#1)


---

## #1

### 알고 있는 metric에 대해 설명해주세요. (ex. RMSE, MAE, recall, precision ...)

**Recall**은 GT 중에서 얼마나 재현했는가**(TP/TP+FN)**,  
**Precision**은 예측한 값 중에서 GT를 얼마나 표현했는가**(TP/TP+FP)**를 말합니다.  
**RMSE, MAE는 각 loss를 구하는 함수**로 classification이 아닐 경우 metric으로 사용할 수 있습니다.

![](https://images.velog.io/images/hanlyang0522/post/5fbfc2fc-f929-49d5-a1c4-7081a44299c3/image.png)
![](https://images.velog.io/images/hanlyang0522/post/a1583180-6c7d-424b-b5ec-01430f675c29/image.png)

<br/>

## #2

### 정규화를 왜 해야할까요? 정규화의 방법은 무엇이 있나요?

데이터의 scale이 크게 차이가 나면 **scale 자체도 feature**라고 생각하고 학습하기 때문에 각 데이터의 크기를 비슷한 scale로 normalize해줍니다.

<br/>

## #3

### Local Minima와 Global Minima에 대해 설명해주세요.

Local minima는 경사하강법(gradient descent)을 사용할 때 경사도가 0이지만 최소점은 아닌 지점, global minima는 경사하강법을 사용할 때 경사도가 0이고 최소점인 지점을 말합니다.  
**Local minima 중 가장 낮은 cost function 값을 갖는 지점이 global minima**입니다.

<br/>

## #4

### 차원의 저주에 대해 설명해주세요.

데이터의 차원(feature, label 혹은 고려해야할 정보)이 너무 많아져 **패턴을 찾기 어려워 오히려 model의 성능이 떨어지는 현상**을 말합니다.

![](https://images.velog.io/images/hanlyang0522/post/b90eafb0-de2e-44d5-b867-21385cc59446/image.png)

- 차원을 줄이는 알고리즘 PCA, LDA, LLE, MDS 등을 사용

<br/>

## #5

### Dimension reduction 기법으로 보통 어떤 것들이 있나요?

PCA, LDA, NMF 등 **feature extraction** 기법을 이용합니다.

- Feature Extration: 높은 차원의 raw feature들을 더 필요한 요소로 추출하는 기법
  - PCA(Principal component analysis)
  - LDA(Linear discriminant analysis)
  - NMF(Non-negativ matrix facotrization)
- Feature Selection: 모든 Feature들 중 필요한 것들만 선택하는 기법  
  EDA 보고 사람이 휴리스틱하게 선택
  - Filtering : leave out dimenstions that do not help much
  - Wrapper : use an external heuristic to select dimensions
  - Embedded : put feature selection into the loss function
- 참고: [Feature Selection (Filter Method & Wrapper Method & Embedded Method)](https://wooono.tistory.com/249), [모델 성과 개선](https://wikidocs.net/16599)

<br/>

## #6

### PCA는 차원 축소 기법이면서, 데이터 압축 기법이기도 하고, 노이즈 제거기법이기도 합니다. 왜 그런지 설명해주실 수 있나요?

PCA는 여러 개의 feature 중 **가장 중요한 feature만 선택하는 기법**으로, 결국은 feature의 수를 줄이기 때문에 상기 효과가 발생합니다. PCA는 데이터를 가장 잘 표현할 수 있는 **몇 개의 주성분 vector만 사용**하기 때문에 비교적으로 덜 중요한 noise를 제거하는 역할을 할 수 있습니다.

- 참고: [차원 축소와 PCA (Principal Components Analysis)](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-9-PCA-Principal-Components-Analysis), [주성분분석(PCA)의 이해와 활용](https://darkpgmr.tistory.com/110)

<br/>

## #7

### LSA, LDA, SVD 등의 약자들이 어떤 뜻이고 서로 어떤 관계를 가지는지 설명할 수 있나요?

SVD(특이값 분해)는 A가 m\*n 행렬일 때 m\*m 직교행렬, n\*n 직교행렬, m\*n **직사각형렬의 행렬곱**으로 분해하는 것입니다.  
LSA(잠재 의미 분석)는 절단된 SVD(truncated SVD)를 이용해 **차원을 축소시키고 노이즈를 제거하거나(CV), 설명력이 높은 정보(NLP)**를 남깁니다.
LDA는 LSA의 단점을 개선하여 탄생한 알고리즘으로, **토픽 모델링에 보다 적합**한 알고리즘입니다.

- 직교행렬(orthogonal matrix): 자신과 자신의 전치 행렬(transposed matrix)의 곱 또는 이를 반대로 곱한 결과가 단위행렬(identity matrix)이 되는 행렬
- 대각행렬(diagonal matrix): 주대각선을 제외한 곳의 원소가 모두 0인 행렬
- 참고: [잠재 의미 분석(Latent Semantic Analysis, LSA)](https://wikidocs.net/24949)

<br/>

## #8

### Markov Chain을 고등학생에게 설명하려면 어떤 방식이 제일 좋을까요?

Markov chain은 **어떤 상태에서 다른 상태로 넘어갈 떄, 바로 이전 단계의 상태에만 영향을 받는 확률 과정**을 의미합니다. 

<br/>

## #9

### 텍스트 더미에서 주제를 추출해야 합니다. 어떤 방식으로 접근해 나가시겠나요?

LDA 기법을 활용합니다. 

- 참고: [잠재디리클레할당](https://huidea.tistory.com/130), [LSA와 LDA의 관계](https://bab2min.tistory.com/585)

<br/>

## #10

### SVM은 왜 반대로 차원을 확장시키는 방식으로 동작할까요? SVM은 왜 좋을까요?

Support Vector Machine(SVM)은 **linearly separable한 데이터에 대해서만 작용**합니다. 하지만 실제 데이터는 non-linearly separable, 즉 비선형 SVM이 많습니다. 이러한 **비선형 SVM은 확장**을 통해 SVM으로 만들어줍니다. 데이터를 비선형 매핑(Mapping)을 통해 **고차원으로 변환하기 때문에 데이터를 효과적으로 표현하는 패턴**을 찾아낼 수 있습니다.

<br/>

## #11

### 다른 좋은 머신 러닝 대비, 오래된 기법인 나이브 베이즈(naive bayes)의 장점을 옹호해보세요.

Naive bayes는 supervised learning에서 매우 효율적이며, **학습에 필요한 데이터의 양이 매우 적다**는 장점이 있습니다. 또한 **정확성도 높고 대용량 데이터에 대해 속도도 빠릅니다**.  
하지만 **모든 feature가 독립이어야 한다**는 가정이 필요하기 때문에 사용에 주의가 필요합니다.

<br/>

## #12

### 회귀 / 분류시 알맞은 metric은 무엇일까?

회귀엔 MSE, MAE, R2를 사용하고  
분류엔 accuracy, precision, recall과 이를 조합한 F1 score를 사용합니다.

- Regression
  - RSS: 단순 오차 제곱 합
  - MSE: L2 score, RSS의 평균
  - MAE: L1 score, MSE보다 outlier에 robust
  - RMSE: MSE에 루트 씌워서 outlier
  - R2: 결정계수, 1-(RSS/전체 분산), 회귀 모델의 설명력을 표현하는 지표, 잘 예측할수록 1에 가까워짐
- Classification
  - Accuracy: (TP+TN) / (TP+TN+FP+FN), 분류 결과가 얼마나 True로 나왔느냐
  - Precision: TP/(TP+FP), ‘모델이 P라고 분류한 데이터 중에서’ 실제로 P인 데이터의 비율
  - Recall: TP/(TP+FN), ‘실제 P인 데이터 중에서’ 모델이 P라고 잘 예측한 데이터의 비율
  - FPR: FP/(FP+TN), False Positive Rate, 실제로 N인 데이터 중에서 모델이 P라고 잘못 분류한 데이터의 비율
  - F1 score: 정밀도와 재현율의 가중 조화평균

<br/>

## #13

### 좋은 모델의 정의는 무엇일까요?

데이터의 패턴을 잘 학습하여 outlier를 잘 걸러내고 새로운 data에 대해서도 성능이 좋은 **robust한 모델**이 좋은 모델이라고 생각합니다.

Keyword:  
Answer: Outlier를 잘 걸러내고 데이터의 패턴을 잘 학습해 새로운 data에서도 성능이 좋은 robust한 모델이 좋은 모델이라고 생각합니다.

<br/>

## #14 

### Association Rule의 Support, Confidence, Lift에 대해 설명해주세요.

Support는 전체 경우의 수에서 두 항목이 같이 나오는 비율 **P(X&Y)**,  
Confidence는 X가 나온 경우 중 X와 Y가 함께 나올 비율 **P(X&Y)/P(X)**,  
Lift는 X와 Y가 같이 나오는 비율을 X가 나올 비율과 Y가 나올 비율의 곱으로 나눈 값 **P(X&Y)/P(X)\*P(Y)** 입니다.  
Lift가 1보다 클 경우 positive correlation, 1일때 independent, 1 미만일 경우 negative correlation입니다.

Keyword:  
Answer: Support는 전체 경우의 수에서 두 아이템이 같이 나오는 비율, Confidence는 X가 나온 경우 중 X와 Y가 함께 나올 비율, Lift는 X와 Y가 같이 나오는 비율을 X가 나올 비율과 Y가 나올 비율의 곱으로 나눈 값으로 1보다 높을 때 positive correlation, 1일때 independent, 1 미만일 때 negatively correlation입니다.

- Assiciation rule: 어떤 사건이 얼마나 자주 함께 발생하는지, 서로 얼마나 연관돼있는지 표시하는 것
- 참고: [https://bab2min.tistory.com/585](https://process-mining.tistory.com/34)

<br/>

## #15 

### 최적화 기법중 Newton’s Method와 Gradient Descent 방법에 대해 알고 있나요?

Newton's method: **방정식 f(x)=0의 해를 근사적으로** 찾는 기법  
Gradient Descent: **f'(x) = 0**이 되는 점을 찾는 기법

<br/>

## 머신러닝(machine)적 접근방법과 통계(statistics)적 접근방법의 둘간에 차이에 대한 견해가 있나요?

Keyword: 성공확률, 신뢰도  
Answer: 머신러닝은 예측의 성공확률을 높이는게 목적이고, 통계는 예측의 신뢰도를 중요하게 여깁니다.

- 머신러닝 방법과 전통적 통계분석 방법은 사용 알고리즘, 방법론보다는 사용하는 목적에서 가장 큰 차이가 발생
- 머신러닝 방법은 예측의 성공 확률을 높이는 데에 목적, 따라서 모델의 신뢰도나 정교한 가정은 상대적으로 중요성이 낮아지며, 오버피팅은 어느 정도 감안하더라도 여러 인자를 사용해 예측을 수행한다.
- 전통적 통계분석 방법은 정해진 분포나 가정을 통해 실패 확률을 줄이는 데에 목적, 따라서 모형의 복잡성보다는 단순성을 추구하며, 신뢰도가 중요해진다. 해석 중점

<br/>

## 인공신경망(deep learning이전의 전통적인)이 가지는 일반적인 문제점은 무엇일까요?

Keyword:
Answer: local minima, overfitting 등의 문제를 해결하지 못했다는 단점이 있습니다.

- 인공신경망에서 layer를 3개 이상 쌓으면 deep learning

<br/>

## 지금 나오고 있는 deep learning 계열의 혁신의 근간은 무엇이라고 생각하시나요?

Keyword:
Answer: GPU 등 하드웨어의 발전으로 layer를 많이 쌓을 수 있었던 것이 혁신의 근간이라고 생각합니다.

<br/>

## ROC 커브에 대해 설명해주실 수 있으신가요?

Keyword: TPR, FPR의 비율, 좌상단에 위치할수록 좋음
Answer: 이진분류기에서 true positive rate와 false positive rate의 비율로 그래프를 그린 것으로, ROC 커브가 좌상단에 붙어있을수록 더 좋은 분류기를 의미합니다.

- ROC(Receiver Operating Characteristic) curve는 다양한 threshold에 대한 이진분류기의 성능을 한번에 표시한 것
  이진 분류의 성능은 True Positive Rate와 False Positive Rate 두 가지를 이용해서 표현
  좌상단에 붙어있는 커브가 더 좋은 분류기를 의미

  ![](https://images.velog.io/images/hanlyang0522/post/c9257c30-3948-408f-b716-7b08aa2617e5/image.png)

  ![](https://images.velog.io/images/hanlyang0522/post/108a5f79-e93a-4d41-b9ac-7794849ff5ca/image.png)

- 참고 https://angeloyeo.github.io/2020/08/05/ROC.html

<br/>

## 여러분이 서버를 100대 가지고 있습니다. 이때 인공신경망보다 Random Forest를 써야하는 이유는 뭘까요?

Keyword: Ensemble
Answer: Random forest는 모델을 ensemble하기 때문에 여러 대의 서버에서 나온 결과를 ensemble해서 병렬적으로 처리하기 때문에 서버가 많을 경우 쓰기에 좋음?

- random forest는 ensemble 모델이라 regularization이 잘 돼서?

## K-means의 대표적 의미론적 단점은 무엇인가요? (계산량 많다는것 말고)

Keyword: 초기화 잘못 하면 local minima, 해석 어려움
Answer: K-means clustering은 중심값 선정, 거리로 데이터 분류, 분류 완료까지 반복의 과정을 거치기 때문에 중앙값 초기화를 잘못 할 경우 local minima에 빠질 위험이 있다. 또한 모든 데이터를 거리로만 판단하기 때문에 사전에 주어진 목적이 없어 결과 해석이 어렵다는 단점이 있습니다.

- 분석 결과가 관찰치 사이의 거리 또는 유사성을 어떻게 정의하느냐에 따라 크게 좌우된다. 즉, K-means algorithm은 초기화에 따라 다른 결과가 나타날 수 있다. 초기화가 잘못 된다면 나쁜 경우 local optima에 빠지는 경우가 존재한다. (참고 : https://wikidocs.net/4693)
- 결과해석이 어렵다. (∵탐색적인 분석방법으로 장점을 가지고 있는 반면 사전에 주어진 목적이 없으므로 결과를 해석하는데 어려움이 존재)

<br/>

## L1, L2 정규화에 대해 설명해주세요.

Keyword:
Answer: L1 norm은 두 벡터의 차의 절대값을 합하고 L2는 Euclidean distance를 구합니다. L2 norm은 특정 가중치가 너무 커져 과적합 되는 것을 방지합니다.

- Cross Validation은 무엇이고 어떻게 해야하나요?
- XGBoost을 아시나요? 왜 이 모델이 캐글에서 유명할까요?
