
_ _ _
 # 충남 챗봇 수업의 저장소
_ _ _
 ## 2023_09_04
* 공유에 필요한 정보
* 수업내용
_ _ _
# 0905
* 파이토치 코드 맛보기
* 모델의 네트워크 생성
* model 클래스의 객체 생성
* 모델의 파라미터 정의
* CPU/GPU 사용지정
* 모델 학습
* 테스트 데이터셋으로 모델 예측
* 모델의 예측 확인
* 가장큰 값을 갖는 인덱스 확인
* 테스트 데이터셋을 이용한 정학도 확인
    * 정확도, 재횬율, 정밀도, F1-스코어
* 지도 학습
    * 분류, 회귀
* K-최근접 이웃
_ _ _
# 0906
* 서포트 벡터 머신
* 결정 트리
* 로지틱 회귀
* 선형 회귀
* K-평균 군집화
* 밀도 기반 군집 분석
* 주성분 분석(PCA)
_ _ _
# 0907
* 딥러닝
    *  AND 게이트
    * OR 게이트
    * XOR 게이트
* 딥러닝 용어
    * 입력층
    * 은닉층
    * 출력층
    * 가중치 *
    * 바이어스
    * 가중합, 전달 함수
    * 함수
        * 활성화 함수
            * 시그모이드 함수
            * 하이퍼볼릭 탄젠트 함수
            * 렐루 함수
            * 리키 렐루 함수
            * 소프트맥스 함수
        * 손실 함수
            * 평귱 제곱 오차
            * 크로스 엔트로피 오차
    * 딥러닝 학습
        * 순전파, 역전파
    * 딥러닝의 문제점과 해결 방안
        * 과접합 문제 발생
            * 과소적합, 적정적합, 과접합
        * 신경망
            * 일반적인 신경망, 드롭아웃이 적용된 신경망
        * 경사 하강법, 배치 경사 하강법, 확률적 경사 하강법, 미니 배치 경사 하강법
    * 옵티마이저
    * 특성추출
    * 딥러닝 알고리즘
        * 심층 신경망, 합성 신경망, 순환 신경망, 제한된 볼츠만 머신, 심층 신뢰 신경망
* 합성곱 신경만
    * 합성곱
    * 합성곱 신경망 구조
    * 1D,2D,3D 합성곱
    * 데이터셋
    * 전이 학습
    * 특성 추출 기법
    * 학습에 사용될 이미지 출력
_ _ _
# 9011
* 이미지 출력 2
* 미세 조정 기법
* 설명 가능한 CNN
    * 특성맵 시각화
* 그래프 합성곱 네트워크
    * 그래프 신경망
        * 인접 행렬, 특성 행렬
_ _ _
# 0912
* 이미지 출력 3
* LeNet
* AlexNet
* VGGNet
* R-CNN
설명
* 공간 피라미드 풀링
* Fast R-CNN
* Faster R-CNN
* 이미지 분활을 위한 신경망
* 함성곱 & 역합성곱 네트워크
_ _ _
# 0913
* IP 설명
* 파이썬 플라스크 
* 장고
* 보틀
* FastAPI
* 플라스크 설치
* flask run
* 디버그 모드
* . env 사용법
* 애플리케이션 루트
* 라우팅
* 템플릿 엔진 이용하기
* 애플리케이션 컨텍스트
* 컨텍스트의 라이프 사이클
* 문의 폼 만들기
* PRG 패턴
* 문의 폼의 템플렛 만들기
* Flask 메세지
* 로깅
* 이메일 보내기
* 쿠키
* 세션
* 응답
_ _ _
# 0914
* 데이터베이스를 이용한 앱 만들기
    * 디렉터 구성
    * Blueprint의 이용
    * SQLAlchemy 설정하기
    * 데이터베이스 조작하기
    * 데이터베이스를 사용한 CRUD 앱 만들기
    * 템플릿의 공통화와 상속
    * CONFIG 설정하기
        * 이메일 송수신 기능, 문의 기능, 로그인 기능, 엔드 포인트,
         신규 작성 기능, 사용자 일람 기능, 사용자 편집, 사용자 삭제 기능
_ _ _
# 0918
* 사용자 인증 기능
* 앱에 인증기능 등록하기
* 회원가입 기능 만들기
* 로그인 기능 만들기
* 로그아웃 기능 만들기
* 앱의 사양과 준비
* 물체 감지 앱의 사양
* 디렉터리 구성
* 물체 감지 앱 등록하기
_ _ _
# 0925
* 조기종료를 이용한 성능 최적화
* 자연어 전처리
    * 토큰
    * 토큰화
    * 불용어
    * 어간 추출
    * 품사 태깅
* Det, Noun, Verd, Prep
* VBZ, PRP, JJ, VBG, NNS, CC
* NLYK
    * 멀뮹치, 토큰 생성, 형태소 분석, 품사 태깅
* KoNLPy
* Gensim
* 임베딩, 토픽 모델링, LDA
* 사이킷런
* 전처리
    * 결측치 확인, 처리
* 토큰화
    * 문장 토큰화
    * 단어 토큰화
    * 한글 토큰화 예제
* 불용어 제거
* 어간 추출
* 표제어 추출
* 정규화
_ _ _
# 0926
* RobustScaker()
* MaxAbsScaker()
* 손실함수와 옵티마이저 저장
* 임베딩
    * 희소 표현 기반 임베딩
        * 원-핫 인코딩
    * 횟수 기반 임베딩
        * 카운터 벡터
        * TF-IDF
    * 예측 기반 임베딩
        * 워드투벧터
        * CBOW, skip-gram
        * 패스트텍스트
    * 횟수/예측 가반 임베딩
        * 글로브
* 트랜스포머 어텐션
    * 린코더
    * 디코더
    * seq2seq
_ _ _
# 0927
* 버트
* 한국어 임베딩

_ _ _
# 1004
* 클러스터링
* K-평균 군집화
* 가우시안 혼합 모델
* 자기 조직화 지도
* SOM 구조