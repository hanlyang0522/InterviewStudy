# 📄Table of Contents

- [딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?](#1)
- [Cost Function과 Activation Function은 무엇인가요?](#2)
- [Tensorflow, PyTorch 특징과 차이가 뭘까요?](#3)
- [Data Normalization은 무엇이고 왜 필요한가요?](#4)
- [알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)](#5) 
- [오버피팅일 경우 어떻게 대처해야 할까요](#6)
- [하이퍼 파라미터는 무엇인가요?](#7)
- [Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?](#8)
- [볼츠만 머신은 무엇인가요?](#9)
- [뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가?](#10)

---

## #1 

### 딥러닝은 무엇인가요? 딥러닝과 머신러닝의 차이는?


딥러닝은 **neural network**를 이용해 ML 학습을 수행하는 것으로, ML과의 가장 큰 차이점은 **feature selection**에 사람이 개입하지 않는다는 것입니다

- ML: 휴리스틱하게, 특징선택에서 사람이 개입  
  Input과 output의 상관관계를 파악해 새로운 input에 대한 output을 예측함! 이때 쓰이는게 stаtistiсаl leаrning аlgоrithms
- DL: 학습을 통해 end-to-end, data dependent
- 가장 큰 차이는 feature selection!!

<br/>

## #2

### Cost Function과 Activation Function은 무엇인가요?

Cost function은 모델의 **예측값과 오차를 계산**해주는 함수로 학습 뱡향을 어느 방향으로 얼마나 개선할지 판단하는 지표가 됩니다. Activation function은 **선형 함수를 비선형함수로** 만들어 표현력을 더 키워주는 함수입니다.

- Activation function이 없을 경우 layer를 깊게 쌓아도 선형함수기 때문에 의미가 없다. AF를 이용해 모델의 복잡도를 높이고 비선형 문제를 해결할 수 있게 만드는것.

<br/>

## #3

### Tensorflow, PyTorch 특징과 차이가 뭘까요?
 
Tensorflow는 단일 데이터 흐름 **그래프를 만들고 그래프 코드를 성능에 맞게 최적화한 다음 모델을 학습하므로** 더 쉽게 다른 언어나 모듈에 적용이 가능합니다.   
PyTorch는 **각 반복 단계에서 즉석으로 그래프를 재생성하므로** 모델 그래프를 만들 때 고정 상태가 아니므로 데이터에 따라 조절이 가능한 유연성을 갖고 있습니다.

- Tensorflow: 단일 데이터 흐름 그래프를 만들고 그래프 코드를 성능에 맞게 최적화한 다음 모델을 학습  
  더 쉽게 다른 언어나 모듈에 적용이 가능  
  즉시 실행, 직관적인 고수준 API, 모든 플랫폼에서의 유연한 모델 구축  
  Define and run

- PyTorch: 예전의 토치(Torch) 및 카페2(Caffe2) 프레임워크를 기반  
  파이썬을 스크립팅 언어로 사용하며, 진화된 토치 C/CUDA 백엔드를 사용  
  각 반복 단계에서 즉석으로 그래프를 재생성  
  플라스크(Flask) 웹 서버로 실행  
  모델 그래프를 만들 때 고정 상태가 아니므로 데이터에 따라 조절이 가능한 유연성
  Define by run

<br/>

## #4

### Data Normalization은 무엇이고 왜 필요한가요?
 
Data Normalization는 입력 data의 scale을 비슷한 크기로 맞춰줘 **weight가 scale에 따라 bias되는 것을 방지**합니다.

- $x=\frac{x−x_{min}}{x_{max}−x_{min}}$
- 입력 data의 scale(규모, 폭)에 따라 weight가 bias되는 것을 방지하기 위해 0~1 사이로 값의 scale을 비슷하게 줄여줌
- Train에도 도움됨
- 참고: https://skyil.tistory.com/50

- Standardization: $ z(x) = {x-m\over\delta} $ 를 벗어나는 값을 지워서 ML 모델이 일반적인 경우에 집중해 올바를 결과를 내게 도와줌

<br/>

## #5

### 알고있는 Activation Function에 대해 알려주세요. (Sigmoid, ReLU, LeakyReLU, Tanh 등)

![](https://images.velog.io/images/hanlyang0522/post/345ef903-f85e-43c1-9b42-55886bc18ce0/image.png)

Sigmoid: output값을 **0에서 1사이**로 만들어준다. 데이터의 평균은 0.5를 갖게된다.  
  위 그림에서 시그모이드 함수의 기울기를 보면 알 수 있듯이 input값이 어느정도 크거나 작으면 기울기가 아주 작아진다.  
  이로인해 생기는 문제점은 **gradient vanishing** 현상이 있다.

Tanh: 그림에서 보면 알 수 있듯이 시그모이드 함수와 거의 유사하다.  
  차이는 **-1 ~ 1**값을 가지고 데이터의 평균이 0이라는 점이다. 데이터의 평균이 0.5가 아닌 0이라는 유일한 차이밖에 없지만 대부분의 경우에서 **sigmoid보다 Tanh가 성능이 더 좋다.**  
  --> Tanh의 기울기가 더 높기 때문에 train에서 성능이 더 좋음
  그러나 시그모이드와 마찬가지로 **gradient vanishing**라는 단점이 있다.

ReLU: 대부분의 경우 일반적으로 ReLU의 성능이 가장 좋기때문에 ReLU를 사용한다.  
  대부분의 input값에 대해 **기울기가 0이 아니기 때문에 학습이 빨리 된다.**  
  학습을 느리게하는 원인이 gradient가 0이 되는 것인데 이를 대부분의 경우에서 막아주기 때문에 시그모이드, Tanh같은 함수보다 학습이 빠르다.
  그림을 보면 input이 0보다 작을 경우 기울기가 0이기 때문에 대부분의 경우에서 기울기가 0이 되는것을 막아주는게 납득이 안 될수 있지만 실제로 hidden layer에서 대부분 노드의 z값은 0보다 크기 때문에 기울기가 0이 되는 경우가 많지 않다.  
  단점으로는 위에서 언급했듯이 z가 음수일때 기울기가 0이라는 것이지만 실제로는 거의 무시할 수 있는 수준으로 학습이 잘 되기 때문에 단점이라 할 수도 없다.

leaky ReLU: ReLU와 유일한 차이점으로는 max(0, z)가 아닌 max(0.01z, z)라는 점  
  즉, input값인 z가 음수일 경우 기울기가 0이 아닌 0.01값을 갖게 된다.  
  leaky ReLU를 일반적으로 많이 쓰진 않지만 ReLU보다 학습이 더 잘 되긴 한다.

- 참고: https://ganghee-lee.tistory.com/32

<br/>

## #6

### 오버피팅일 경우 어떻게 대처해야 할까요?

Overfitting이 일어날 경우에는 augmentation를 이용해 **데이터의 절대적인 양을 늘리거나 regularization, dropout을 사용하고 norm penalty를 사용해 weight가 너무 커지지 않게** 만듭니다.

- 오버피팅: train loss는 줄지만, eval loss는 안 줄때
  1. 데이터 양 늘리기  
     데이터의 양이 적을 경우, 해당 데이터의 특정 패턴이나 노이즈까지 쉽게 암기하기 되므로 과적합 현상이 발생할 확률이 늘어남
  2. ~~모델 복잡도 줄이기~~  
     ~~복잡도: 은닉층(hidden layer)의 수나 매개변수의 수~~  
     --> 오히려 모델의 크기를 키우는게 답일수도
  3. Regularization 사용
  4. Dropout 사용
  5. Cost function에 Norm penalty(L1, L2 등) 적용

<br/>

## #7

### 하이퍼 파라미터는 무엇인가요?

Learning rate나 SVM에서의 C, sigma 값, KNN에서의 K값 등 모델링할 때 **데이터에서 추정될 수 없는, 사용자가 직접 세팅해주는 값**을 말합니다.

- A model hyperparameter is a configuration that is external to the model and whose value cannot be estimated from data.


<br/>

## #8

### Weight Initialization 방법에 대해 말해주세요. 그리고 무엇을 많이 사용하나요?

**Gaussian sampling**을 하거나, 이전 node 수에 영향을 받는 **LeCun init**, 다음 node 수에도 영향을 받고 uniform/normal 분포를 따르는 **Xavier init, He init** 등이 있다.  
Activation function이 sigmoid, tanh일 경우 Xavier init을, ReLU일 경우 He init을 사용함.


<br/>

## #9

### 볼츠만 머신은 무엇인가요?

볼츠만 머신(Boltzmann machine)은 볼 수 있는 층(visible layer)과 숨겨진 층(hidden layer)의 **두 층으로 구성된 모델**입니다. Layer의 모든 node들은 연결되어 있습니다.

- 볼츠만 머신(Boltzmann machine)은 볼 수 있는 층(visible layer)과 숨겨진 층(hidden layer)의 두 층으로 구성된 그래픽 모델
- 이 모델에서 각 볼 수 있는 유닛은 숨겨진 유닛들과만 연결되고, 볼 수 있는 유닛들 사이에 그리고 숨겨진 유닛들 사이에는 서로 연결이 없을 때 이를 제한된 볼츠만 머신(RBM: restricted Boltzmann machine)이라 한다


<br/>

## #10

### 뉴럴넷의 가장 큰 단점은 무엇인가? 이를 위해 나온 One-Shot Learning은 무엇인가?

Neural network의 가장 큰 단점은 **data의 수가 적을 경우 사용하지 못한다는 점**입니다.  
One-Shot Learning은 이를 개선하기 위해 각 class마다 1개의 image만 사용하고 input image를 **각 class별 거리를 측정해** 가장 가까운 class로 판정하는 기법입니다.


<br/>

## #11 

### 요즘 Sigmoid 보다 ReLU를 많이 쓰는데 그 이유는?

Sigmoid는 데이터의 평균이 0.5로 수렴하고 input의 크기가 어느정도 크거나 작으면 기울기가 매우 작아져 **gradient vanishing** 문제가 발생합니다.   
반면에 ReLU는 대부분의 input 값에 대해 기울기가 0이 아니고 input을 그대로 output으로 내보내기 때문에 **학습이 빨리** 됩니다. ReLU도 input이 0보다 작을 경우엔 기울기가 0이 되지만 hidden layer에서 node의 z값은 대부분 0보다 크기 때문에 **학습이 잘 되는 편**입니다.

<br/>

## #12

### Non-Linearity라는 말의 의미와 그 필요성은?

Non-linearity는 **비선형, 즉 입력과 출력이 비례하지 않는다**는 것을 뜻합니다.
데이터가 복잡해지고 **feature의 차원이 증가**하면서 단순한 선형의 boundary로는 표현이 불가능해졌기 때문에 nonlinearity가 필요해졌습니다.

<br/>

## #13

### ReLU로 어떻게 곡선 함수를 근사하나?

ReLU는 **선형(y=x)과 비선형(y=0)의 결합**이기 때문에 ReLU가 반복해 적용되면 선형부분의 결합으로 곡선 함수를 표현할 수 있습니다.

<br/>

## #14

### ReLU의 문제점은?

**Input이 0보다 작을 경우** gradient가 0이 됩니다.

<br/>

## #15

### Bias는 왜 있는걸까?

Model이 항상 원점을 기준으로 분포해있지는 않기 때문에 model이 data에 잘 fit할 수 있게 **평행이동**시켜줍니다.

<br/>

## #16

### Gradient Descent에 대해서 쉽게 설명한다면?

함수의 **기울기(gradient)를 이용해 함수의 최소값을 찾아가는 iterative한 방법**을 말합니다.

<br/>

## #17

### GD 중에 때때로 Loss가 증가하는 이유는?

**Learning rate**가 커서 순간적으로 발산하거나, local minimum을 빠져나오기 때문입니다.

<br/>

## 18

### Back Propagation에 대해서 쉽게 설명 한다면?

Cost function을 이용해 Ground truth와 estimate output의 차이가 얼마나 나는지 구한 후 **loss만큼 weight를 업데이트**합니다.

<br/>

## #19 

### Local Minima 문제에도 불구하고 딥러닝이 잘 되는 이유는?

Lecun, Xavier, He init 등 weight를 단순히 zero initialize하는 것이 아니라 **activation function에 따라 적절한 초기화 방법**이 등장했고, pre-trained model을 이용하여 효과적으로 학습할 수 있기 때문입니다.

<br/>

## #20

### Gradient descent가 Local Minima 문제를 피하는 방법은?

LambdaLR, CosineAnnealingLR 등 **learning rate scheduler**를 이용해 주기적으로 lr의 크기를 변경하여 local minima를 빠져나갈 수 있게 합니다.

<br/>

## #21

### 찾은 해가 Global Minimum인지 아닌지 알 수 있는 방법은?

차원이 높을수록 local minima에 빠질 위험이 적기 때문에 차원을 높여서 실험하고 seed를 변경하며 **여러번 실험**을 반복합니다.

<br/>

## #22

### Training 세트와 Test 세트를 분리하는 이유는?

Train에서 test 세트를 학습하면 모델이 추정하는 분포가 test에 **overfit**하여 unseen data에 대한 성능이 떨어질 수 있기 때문입니다.

<br/>

## #23

### Validation 세트가 따로 있는 이유는?

Unseen data에 대한 **모델의 성능을 평가하기 위해서** 사용합니다.

<br/>

## #24

### Test 세트가 오염되었다는 말의 뜻은?

Test용으로 분리해야하는 데이터가 train 데이터에 반영됐을 경우를 뜻합니다.

<br/>

## #25

### Regularization이란 무엇인가?

L1, L2 regularization처럼 **모델에 penalty**를 줘 train data에 대한 perfect fit을 포기하는 대신 testing accuracy를 높이고자 하는 기법입니다.


<br/>

## #26

### Batch Normalization의 효과는?

Gradient의 스케일이나 초기 값에 대한 dependency 가 줄어들어 large learning rate 를 설정할 수 있기 때문에 **빠른 학습이 가능**하며, **gradient vanishing을 방지**할 수 있습니다.. 또한 scale, shift 변환을 통해 nonlinearity를 유지하며 학습하여 **regularization 효과**가 있습니다.

- 참고: https://eehoeskrap.tistory.com/430
  https://m.blog.naver.com/laonple/220808903260

<br/>

## #27

### Dropout의 효과는?

학습 데이터에 의해 각 node들이 **co-adaptation**되는 현상을 방지하고 여러 개의 모델을 학습시키는 것과 같은 작용을 하여 **regularization**효과를 기대할 수 있습니다.

<br/>

## #28

### BN 적용해서 학습 이후 실제 사용시에 주의할 점은? 코드로는?

Inference 시 input을 이용해 BN을 하면 모델이 train에서 input의 분포를 추정한 의미가 없어지기 때문에, inference 시에는 결과를 deterministic하게 만들기 위해 미리 저장해둔 **mini-batch의 moving average를 이용**해 정규화를 합니다.

<br/>

## #29

### GAN에서 Generator 쪽에도 BN을 적용해도 될까?

DCGAN의 generator의 마지막 layer와 discriminator의 첫번쨰 layer는 모델이 distribution의 정확한 mean, scale을 학습하기 위해 BN을 적용하지 않지만, 그 외 **대부분의 layer에는 BN을 적용**합니다.  
하지만 **mini-batch의 크기가 작을 경우** GAN이 생성하는 image가 z보다 BN의 변동값에 영향을 많이 받아 **batch간 correlation이 생기는 문제**가 발생하기도 합니다.

  - Z code: 생성자는 랜덤 벡터 ‘z’를 입력으로 받아 가짜 이미지를 출력하는 함수다. 여기서 ‘z’는 단순하게 균등 분포(Uniform Distribution)나 정규 분포(Normal Distribution)에서 무작위로 추출된 값이다.
  - 참고: https://kakalabblog.wordpress.com/2017/07/27/gan-tutorial-2016/

<br/>

## #30

### SGD, RMSprop, Adam에 대해서 아는대로 설명한다면?

![](https://images.velog.io/images/hanlyang0522/post/807cea1e-b014-42e2-9be0-156a7b9abcd2/image.png)

SGD는 loss function을 계산할 때 전체 batch가 아니라 **일부 데이터(mini-batch)만 사용해 같은 시간에 더 많은 step**을 갈 수 있어 local minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성이 높다.  
RMSprop은 기울기를 단순 누적하지 않고 weighted moving average를 사용해 **최신 기울기를 더 크게 반영**하도록 한다.   
Adam은 **momentum과 RMSprop을 결합해 스텝 방향과 사이즈 모두 고려**함.

### SGD에서 Stochastic의 의미는?

Keyword: 확률론적인  
Answer: 확률론적인, 약간은 랜덤한 결과, 약간의 불확실성을 포함한 다양한 프로세스를 의미

### 미니배치를 작게 할때의 장단점은?

Keyword: 연산량을 줄임  
Answer: 연산량을 줄여 같은 시간에 더 많은 step을 나갈 수 있다

### 모멘텀의 수식을 적어 본다면?

![](https://images.velog.io/images/hanlyang0522/post/03b9292b-5ea4-494e-923b-69bcd0f6d7a0/image.png)

Keyword: 관성을 적용  
Answer: W를 업데이트 할 때 이전 단계의 업데이트 방향을 반영  
a는 learning rate, m은 모멘텀 계수

- 간단한 MNIST 분류기를 MLP+CPU 버전으로 numpy로 만든다면 몇줄일까?
  - 어느 정도 돌아가는 녀석을 작성하기까지 몇시간 정도 걸릴까?
  - Back Propagation은 몇줄인가?
  - CNN으로 바꾼다면 얼마나 추가될까?
https://github.com/hanlyang0522/InterviewStudy.githttps://github.com/hanlyang0522/InterviewStudy.git