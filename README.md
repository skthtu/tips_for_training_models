## モデル学習時のTips集

### １.Mixed Precision Training (混合精度学習)
  #### 概要：値計算の際に単精度と半精度を混ぜて計算することで、メモリの節約と計算の高速化を図る手法。
  ##### メリット： メモリ消費量と学習時間が減る。 (学習時間は2分の１くらいになっている。)
  ##### デメリット: 精度が若干落ちる可能性がある。(使っている感じ、精度はほとんど変わらず、学習時間の削減を考えると使った方がいいと思う。)
  
  ちなみにtrainerを使うときは、training_argsで "fp16 = True" とする。
  
  参考:<br>
  ・[Pytorch 公式ドキュメント](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training)<br>
  ・[AI4CodeのBaseline](https://github.com/skthtu/ai4code-baseline/blob/main/code/train.py)<br>
  ・[AI SHIFTの記事](https://www.ai-shift.co.jp/techblog/2138)<br>
  ・[俵さんの記事](https://tawara.hatenablog.com/entry/2021/05/31/220936)<br>
  
  混合精度学習を使ったtrain↓ (Pytorch公式ドキュメントより引用)
    
    # Creates model and optimizer in default precision
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            # Runs the forward pass with autocasting.
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(loss).backward()

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
          
  ちなみに通常のtrain↓

    # Creates model and optimizer in default precision
    model = Net().cuda()
    optimizer = optim.SGD(model.parameters(), ...)

    for epoch in epochs:
        for input, target in data:
            optimizer.zero_grad()

            output = model(input)
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()


### 2.Gradient Accumulation (勾配累積)
  #### 概要: 小さいバッチで計算した勾配を数ステップ累積しておき、その平均をとって、パラメータ更新を行う手法。補足だが、平均をとっていないcodeも散見されどちらが正しいか不明。
  ###### メリット： 実際のバッチサイズより大きいバッチサイズで学習した場合と同じ効果を期待できる。
  
  参考：<br>
  ・[AI4CodeのBaseline](https://github.com/skthtu/ai4code-baseline/blob/main/code/train.py)←平均していない<br>
  ・[AI SHIFTの記事](https://www.ai-shift.co.jp/techblog/2138)<br>
  
  混同精度学習を使った場合は以下。


    if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1: #accumulation_stepsごとに更新
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()


### 3.Data Parallel
  #### 概要: 単一のマシンに搭載された複数のGPUを使って並行して計算を行えるようにする。
  ###### メリット： 計算の高速化。
  
  参考：<br> 
  ・[Pytorchの公式ドキュメント](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)<br>
  ・[PyTorchでGPUを並列で使えるようにするtorch.nn.DataParallelのメモ from Qiita@m__k](https://qiita.com/m__k/items/87b3b1da15f35321ecf5)<br>
  
  実装
  
    model = simple_model()
    model = torch.nn.DataParallel(model, device_ids=[0, 1]) #device_idsに使用するGPUを指定。GPUに割り当てられているidsは"nvidia-smi"で確認できる。
    model.cuda() #GPUに送信
  
    torch.save(model.module.state_dict(), output_model_path) #保存するときなど、元々のモデルを操作するときは、Dataparallelオブジェクトに対して.moduleを介して操作する。
  
  
　　　　
  
