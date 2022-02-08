# 데이터

https://www.kaggle.com/imoore/60k-stack-overflow-questions-with-quality-rate

Balanced data이다.

Huggingface의 electra 모델을 사용하여 Classification 진행.

# 실험

![image](https://user-images.githubusercontent.com/61371150/153041439-2beeffdd-114e-4448-8cc8-3aaf3a01bd83.png)

<img width="770" alt="스크린샷 2022-02-09 오전 2 23 44" src="https://user-images.githubusercontent.com/61371150/153041511-86bf79d0-1978-4405-a886-b2abb273aff9.png">

•ELECTRA 모델을 여러 조건에서 학습한 후 성능 비교를 통해 파라미터 결정
•모델 구조 PyTorch로 구현.
•Max Length = 256, Learning Rate = 1e-4, Batch Size = 16일 때가 결과가 가장 좋았음.

•Best Accuracy :  92.81%

# 결론

3가지 모델에서 데이터 전 처리, 하이퍼 파라미터 변경, 모델 구조 변경을 통해 분류 정확도를 높이기 위해 실험을 해봤음. 
테스트 데이터 12000개에 대해서 동일하게 테스트를 진행했고 Mutli Class(3)
분류이며 balanced data이기 때문에 f1 score가 아닌 정확도로 모델의 성능을 비교했음.
모델의 구조를 PyTorch로만 구현했을 때, 동일한 조건에서 ELECTRA가 DistilBERT보다 성능이 좋았음.
