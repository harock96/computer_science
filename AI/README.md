#데이터

Kaggle : https://www.kaggle.com/mlg-ulb/creditcardfraud



불균형데이터

# ANN 구현 

![image](https://user-images.githubusercontent.com/61371150/153042848-17ecea4b-8c17-4d8b-a5fe-738e54a1c227.png)

hidden layer : 1 / hidden layer node : 45 / 활성화 함수 : relu / 출력층 활성화 함수 : sigmoid / 
손실함수: binary crossentropy / optimzer : adam / epoch : 5 / batch size : 32 
Resampling : SMOTE / Scaling : normalizer 

![image](https://user-images.githubusercontent.com/61371150/153042923-cd6f19ba-0a66-46ba-a7ce-3cd69026825f.png)
![image](https://user-images.githubusercontent.com/61371150/153042929-c7846015-9c39-4f09-a45b-32a606e67b35.png)

recall / precision / f1score : epoch가 약 2~3일때 이후로는 과대적합 혹은 과소적합이 없는 추이를 보임
loss : epoch 5 이후로는 과대적합이 일어나고 있음

# 지표

주어진 데이터는 정상거래 데이터가 99%이상 / 사기거래가 1% 미만의 불균형 데이터

모든 거래가 정상이라도 판단을 하여도 정확도가 99% 이상이기 때문에 

recall/ precisoin/ f1score 등의 지표 사용필요 

# 결론

Test dat를 그대로 사용했을 경우

![image](https://user-images.githubusercontent.com/61371150/153043250-dde2b4f3-5b98-4d43-b77f-88ff32913351.png)

Test data를 Resampling한 경우

![image](https://user-images.githubusercontent.com/61371150/153043314-c16fd023-fe38-4409-80bf-d326817dd78c.png)

epoch가 5이상일 때

0.95 ~ 0.96





