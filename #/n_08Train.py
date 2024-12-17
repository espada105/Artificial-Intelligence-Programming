# %%time  # 셀 실행에 걸린 시간을 출력 (Jupyter Notebook의 매직 명령어)
if TRAINING:  # TRAINING이 True인 경우에만 학습 실행
    # 변수 초기화
    best_iou_score = 0.0  # 가장 높은 IoU 점수를 저장할 변수
    train_logs_list, valid_logs_list = [], []  # 학습 및 검증 로그를 저장할 리스트 초기화
    for i in range(0, EPOCHS):  # 0부터 EPOCHS까지 학습을 반복
        # Perform training & validation
        print("\nEpoch: {}".format(i))  # 현재 에폭 번호 출력
        train_logs = train_epoch.run(train_loader)  # 훈련 루프 실행 및 로그 반환
        valid_logs = valid_epoch.run(valid_loader)  # 검증 루프 실행 및 로그 반환
        
        train_logs_list.append(train_logs)  # 훈련 로그를 리스트에 저장
        valid_logs_list.append(valid_logs)  # 검증 로그를 리스트에 저장
        # Save model if a better val IoU score is obtained
        if best_iou_score < valid_logs['iou_score']:  # 검증 IoU 점수가 최고 점수보다 높은 경우
            best_iou_score = valid_logs['iou_score']  # 최고 점수를 갱신
            torch.save(model, './best_model.pth')  # 모델을 파일로 저장
            print('Model saved!')  # 모델 저장 메시지 출력
