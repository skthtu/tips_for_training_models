    #train部分のコードのみ抜き出し↓
  
    #optimrizer:
    #To reproduce BertAdam specific behavior set correct_bias=False
    #Whether or not to correct bias in Adam (for instance, in Bert TF repository they use False).
                                           
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, correct_bias=False)  
       
    #scheduler:
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * num_train_optimization_steps,
                                                num_training_steps=num_train_optimization_steps)

    #ここではL1Loss
    criterion = torch.nn.L1Loss() #Loss関数
    
    #混合精度学習Scaler
    scaler = torch.cuda.amp.GradScaler() 

    #trainループ
    for e in range(epochs):
        model.train()
        tbar = tqdm(train_loader, file=sys.stdout)
        loss_list = []
        preds = []
        labels = []

        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
                    
            with torch.cuda.amp.autocast():#混合精度学習
                pred = model(*inputs)
                loss = criterion(pred, target)
            scaler.scale(loss).backward() #逆伝播
            
            #勾配累積をする場合
            if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1: #accumulation_stepsごとに予測を抜ける and 最後に抜ける。
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
