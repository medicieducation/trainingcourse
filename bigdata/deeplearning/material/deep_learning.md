

- AI 소개 : [deep_learning_intro.pptx](../material/deep_learning/deep_learning_intro.pptx)
    - 딥러닝 학습 원리
    - perceptron, DNN
    - 문제 복잡도와 모델 크기
    - AI, ML, DL 관계
    - 다양한 기술적 용어들
- 환경
    - Conda, Jupyter
    - Colab : https://colab.research.google.com/
- 필요 라이브러리들 [libray_for_deep_learning.md](library_for_deep_learning.md)
- Keras를 사용한 DNN : [dnn_in_keras.ipynb](../material/deep_learning/dnn_in_keras.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/dnn_in_keras.ipynb)
    - 기본 구조 : compile, fit, evaluate, predict
    - 데이터 준비. train, test
    - 로스 그래프 그리기
    - 입력 출력 모양과 모델 구조
    - 에폭, batch_size와 성능
    - shuffle, optimier 적용
- 성능
    - 오버피팅 개요
    - 오버피팅 완화법 : [dnn_in_keras_overfitting.ipynb](../material/deep_learning/dnn_in_keras_overfitting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/dnn_in_keras_overfitting.ipynb)
    - callback : [dnn_in_keras_callback.ipynb](../material/deep_learning/dnn_in_keras_callback.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/dnn_in_keras_callback.ipynb)
- 성능 측정
    - 지표 종류
    - 보고서 작성을 위한 AUC, ROC, confusion matrix : [roc_auc_confusion_matric.ipynb](../material/deep_learning/roc_auc_confusion_matric.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/roc_auc_confusion_matric.ipynb)
- 데이터
    - 기본 작업(회귀와 분류)과 활용 작업
    - 데이터 종류와 작업
        - 속성 데이터 : [template_attribute_data_classification.ipynb](../material/deep_learning/template_attribute_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_attribute_data_classification.ipynb)
        - 영상 데이터 : [cnn_mnist.ipynb](../material/deep_learning/cnn_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/cnn_mnist.ipynb)
        - 순차 데이터 : [template_numeric_sequence_data_prediction.ipynb](../material/deep_learning/template_numeric_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_numeric_sequence_data_prediction.ipynb)
        - 자연어 데이터 : [template_word_sequence_data_prediction.ipynb](../material/deep_learning/template_word_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_word_sequence_data_prediction.ipynb)
    - 데이터 전처리 4가지 : [flawed_iris_preprocessing.ipynb](library/flawed_iris_preprocessing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/library/flawed_iris_preprocessing.ipynb)
    - Data Generator : [custom_data_generator.ipynb](../material/deep_learning/custom_data_generator.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/custom_data_generator.ipynb)
    - ImageDataGenerator : [data_augmentation_using_ImageDadtaGenerator.ipynb](../material/deep_learning/data_augmentation_using_ImageDadtaGenerator.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/data_augmentation_using_ImageDadtaGenerator.ipynb) 
    - Sequence DataGenerator
        - [financial_data_predict_commodity_price.ipynb](../material/deep_learning/financial_data_predict_commodity_price.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/financial_data_predict_commodity_price.ipynb) 
        - [weather_forecasting.ipynb](../material/deep_learning/weather_forecasting.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/weather_forecasting.ipynb)        
    - TensorFlow DataSet : [tensorflow_data_tfds.ipynb](../material/deep_learning/tensorflow_data_tfds.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/tensorflow_data_tfds.ipynb)
- 속성 데이터
    - 예측 : [template_attribute_data_regression.ipynb](../material/deep_learning/template_attribute_data_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_attribute_data_regression.ipynb)
    - 분류 : [template_attribute_data_classification.ipynb](../material/deep_learning/template_attribute_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_attribute_data_classification.ipynb)
    - 2진 분류 : [template_attribute_data_binary_classification.ipynb](../material/deep_learning/template_attribute_data_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_attribute_data_binary_classification.ipynb)    
- 영상 데이터
    - 영상 데이터의 이해
    - CNN 소개
    - 영상 데이터 증강 : [data_augmentation_using_ImageDadtaGenerator.ipynb](../material/deep_learning/data_augmentation_using_ImageDadtaGenerator.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/data_augmentation_using_ImageDadtaGenerator.ipynb) 
    - crop and resize : [image_crop_and_resize.ipynb](../material/deep_learning/image_crop_and_resize.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/image_crop_and_resize.ipynb) 
    - 예측 - vanilla CNN : [template_image_data_vanilla_cnn_regression.ipynb](../material/deep_learning/template_image_data_vanilla_cnn_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_image_data_vanilla_cnn_regression.ipynb)
    - 예측 - 전이학습 : [template_image_data_transfer_learning_regression.ipynb](../material/deep_learning/template_image_data_transfer_learning_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_image_data_transfer_learning_regression.ipynb)
    - 분류 - vanilla CNN : [template_image_data_vanilla_cnn_classification.ipynb](../material/deep_learning/template_image_data_vanilla_cnn_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_image_data_vanilla_cnn_classification.ipynb)
    - 분류 - 전이학습 : [template_image_data_transfer_learning_classification.ipynb](../material/deep_learning/template_image_data_transfer_learning_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_image_data_transfer_learning_classification.ipynb)
    - 2진 분류 - vanilla CNN : [template_image_data_vanilla_cnn_binary_classification.ipynb](../material/deep_learning/template_image_data_vanilla_cnn_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_image_data_vanilla_cnn_binary_classification.ipynb)
    - 2진 분류 - 전이학습 : [template_image_data_transfer_learning_binary_classification.ipynb](../material/deep_learning/template_image_data_transfer_learning_binary_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_image_data_transfer_learning_binary_classification.ipynb)
- 순차열 데이터
    - 순차열 데이터의 이해
    - RNN 소개
    - 숫자열
        - 단일 숫자열 예측 : [template_numeric_sequence_data_prediction.ipynb](../material/deep_learning/template_numeric_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_numeric_sequence_data_prediction.ipynb)
        - 단일 숫자열 분류 : [template_numeric_sequence_data_classification.ipynb](../material/deep_learning/template_numeric_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_numeric_sequence_data_classification.ipynb)
        - 다중 숫자열 분류 : [template_multi_numeric_sequence_data_classification.ipynb](../material/deep_learning/template_multi_numeric_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_multi_numeric_sequence_data_classification.ipynb) 
        - 다중 숫자열 다중 예측 : [template_multi_numeric_sequence_data_multi_prediction.ipynb](../material/deep_learning/template_multi_numeric_sequence_data_multi_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_multi_numeric_sequence_data_multi_prediction.ipynb)
        - 다중 숫자열 단일 예측 : [template_multi_numeric_sequence_data_one_prediction.ipynb](../material/deep_learning/template_multi_numeric_sequence_data_one_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_multi_numeric_sequence_data_one_prediction.ipynb)
    - 문자열
        - 문자열 예측 : [template_text_sequence_data_prediction.ipynb](../material/deep_learning/template_text_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_text_sequence_data_prediction.ipynb)
        - 문자열 분류 : [template_text_sequence_data_classification.ipynb](../material/deep_learning/template_text_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_text_sequence_data_classification.ipynb)
        - 문자열 연속 예측 : [template_text_data_sequential_generation.ipynb](../material/deep_learning/template_text_data_sequential_generation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_text_data_sequential_generation.ipynb)
- 자연어 데이터
    - 자연어 데이터의 이해
    - 단어열 분류 : [template_word_sequence_data_classification.ipynb](../material/deep_learning/template_word_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_word_sequence_data_classification.ipynb)
    - 단어열 예측 : [template_word_sequence_data_prediction.ipynb](../material/deep_learning/template_word_sequence_data_prediction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_word_sequence_data_prediction.ipynb)
    - 한글 단어열 분류 : [template_korean_word_sequence_data_classification.ipynb](../material/deep_learning/template_korean_word_sequence_data_classification.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/template_korean_word_sequence_data_classification.ipynb)
    - Bert를 사용한 한글 문장 간 관계 분류 : [korean_sentence_relation_classification_with_bert.ipynb](../material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/korean_sentence_relation_classification_with_bert.ipynb)
    - Bert를 사용한 한글 문장 간 관계값 예측 : [korean_sentence_relation_regression_with_bert.ipynb](../material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/korean_sentence_relation_regression_with_bert.ipynb)
- 작업 별
    - 물체 탐지
        - 데이터 준비와 레이블링
        - 학습 실행과 사용
            -  학습 : [train_yolov3_raccoon_train.ipynb](train_yolov3_raccoon_train.ipynb)   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/master/material/deep_learning/train_yolov3_raccoon_train.ipynb)
            - 탐지 실행 : [keras_yolov3_custom_model.ipynb](keras_yolov3_custom_model.ipynb)   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/master/material/deep_learning/keras_yolov3_custom_model.ipynb)

    - 영상 영역 분할
        - U-Net을 사용한 영상 분할 : [unet_segementation.ipynb](../material/deep_learning/unet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/unet_segementation.ipynb)
            - U-Net을 사용한 영상 분할 실습 - 거리 영상 : [unet_setmentation_practice.ipynb](../material/deep_learning/unet_setmentation_practice.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/unet_setmentation_practice.ipynb)
            - U-Net을 사용한 영상 분할 실습 - MRI : [MRI_images.zip](https://github.com/dhrim/deep_learning_data/raw/master/MRI_images.zip)        
        - M-Net을 사용한 영상 분할 : [mnet_segementation.ipynb](../material/deep_learning/mnet_segementation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/mnet_segementation.ipynb)
        - U-Net을 사용한 컬러 영상 분할 : [unet_segementation_color_image.ipynb](../material/deep_learning/unet_segementation_color_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/unet_segementation_color_image.ipynb)      

    - 추천
        - 추천 기반 원리 : [recommendation.ipynb](../material/deep_learning/recommendation.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/recommendation.ipynb) 
        - TensorFlow Recommenations : [TFRS_recommendation_template.ipynb](../material/deep_learning/TFRS_recommendation_template.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/TFRS_recommendation_template.ipynb)
        - 영상 검색 : [image_search_by_ConvAutoEncoder.ipynb](../material/deep_learning/image_search_by_ConvAutoEncoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/image_search_by_ConvAutoEncoder.ipynb)  
        - 소리 검색 : [sound_search_urban_sound.ipynb](../material/deep_learning/sound_search_urban_sound.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/sound_search_urban_sound.ipynb)
    - 이상탐지
        - 노이즈 제거 : [denoising_autoencoder.ipynb](../material/deep_learning/denoising_autoencoder.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/denoising_autoencoder.ipynb)
        - 이상탐지 기반 원리
        - 속성 데이터 이상 탐지 : [financial_data_detect_fraud_card.ipynb](../material/deep_learning/financial_data_detect_fraud_card.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/financial_data_detect_fraud_card.ipynb) 
        - 영상 데이터 이상 탐지 : [anomaly_detection_fahsion_mnist.ipynb](../material/deep_learning/anomaly_detection_fahsion_mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/anomaly_detection_fahsion_mnist.ipynb) 

    - 포즈 추출 : [open_pose_using_template.ipynb](../material/deep_learning/open_pose_using_template.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/medicieducation/trainingcourse/blob/main/bigdata/deeplearning/material/deep_learning/open_pose_using_template.ipynb)
    - 스타일 변환 : https://www.tensorflow.org/tutorials/generative/style_transfer?hl=ko
- 기타
    - GAN
    - 강화 학습
    - 알파고의 이해






















