## モデル学習時のTips集

### １.Mixed Precision Training (混合精度学習)
  #### メリット： メモリ消費量と学習時間が減る。 (学習時間は2分の１くらいになっている。)
  #### デメリット: 精度が若干落ちる可能性がある。(使っている感じ、精度はほとんど変わらず、学習時間の削減を考えると使った方がいいと思う。)
  
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


### 2.Mixed Precision Training (混合精度学習)
  #### メリット： メモリ消費量と学習時間が減る。 (学習時間は2分の１くらいになっている。)
  #### デメリット: 精度が若干落ちる可能性がある。(使っている感じ、精度はほとんど変わらず、学習時間の削減を考えると使った方がいいと思う。)

