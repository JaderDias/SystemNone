// The video's scene markup, in source-time order (timeline.js plays the scenes in its own order).
document.body.insertAdjacentHTML("afterbegin", `
<div id="stage">

  <!-- global auras -->
  <div id="auraA" class="aura a" style="background: radial-gradient(circle, rgba(56,189,248,.55) 0%, rgba(129,140,248,.25) 35%, transparent 65%);"></div>
  <div id="auraW" class="aura a" style="background: radial-gradient(circle, rgba(255,106,61,.45) 0%, rgba(255,55,95,.18) 35%, transparent 65%);"></div>
  <div id="auraG" class="aura a" style="background: radial-gradient(circle, rgba(48,209,88,.45) 0%, rgba(99,230,190,.18) 35%, transparent 65%);"></div>

  <!-- ============ S1: a question ============ -->
  <div class="center"><div id="s1card" class="qcard a">
    <div class="qlabel">Your customer asks</div>
    <div class="qtext"><span id="s1text"></span><span id="s1caret" class="caret"></span></div>
    <div class="qchips"><span class="chip" id="s1c0">refund</span><span class="chip" id="s1c1">cancel</span><span class="chip" id="s1c2">other</span></div>
    <div class="qthink" id="s1think"><span class="spin" id="s1spin"></span><span id="s1status">Asking a frontier model…</span><span class="timer tnum" id="s1timer">0.0 s</span></div>
  </div></div>
  <div class="center" style="top: 300px"><div class="words h2">
    <span id="s1w0" class="a">Slow.</span><span id="s1w1" class="a">Expensive.</span><span id="s1w2" class="a"><span class="grad-warm">Every single time.</span></span>
  </div></div>

  <!-- ============ S2: the bill ============ -->
  <div class="center">
    <div id="s2count" class="mega tnum a">1</div>
    <div id="s2lab" class="h3 gray a" style="margin-top: 10px">message</div>
  </div>
  <div class="center" style="top: -40px">
    <div class="stats">
      <div class="stat a" id="s2a"><div class="big"><span class="grad-warm">$8,120</span></div><div class="lab">per million messages</div></div>
      <div class="stat a" id="s2b"><div class="big"><span class="grad-warm">2.4 s</span></div><div class="lab">for every answer</div></div>
    </div>
  </div>
  <div class="center" style="top: 300px"><div id="s2line" class="h3 a">Just to pick one label.</div></div>
  <div id="s2foot" class="foot a">Claude Opus 5.5 on banking77, a 77-label customer-support benchmark.</div>

  <!-- ============ S3: the insight ============ -->
  <div class="center"><div id="s3q" class="h1 a">What if a <span class="grad-green">tiny</span> model<br>could answer instead?</div></div>
  <div id="s3dc" class="zl"></div>
  <div id="s3rack" class="zl"></div>
  <div id="s3board" class="zl"></div>
  <div id="s3cpu" class="zl"></div>
  <div class="center" style="top: 370px"><div id="s3c1" class="h3 a">A <span style="color: var(--orange)">frontier model</span> trains on a whole datacenter.</div></div>
  <div class="center" style="top: 370px"><div id="s3c2" class="h3 a">It runs on multiple 8 GPU servers.</div></div>
  <div class="center" style="top: 370px"><div id="s3c3" class="h3 a">A <span style="color: var(--cyan)">System One model</span> requires multiple GPUs.</div></div>
  <div class="center" style="top: 370px"><div id="s3c4" class="a"><div class="h3">A <span style="color: var(--green)">student model</span> trains and runs on one CPU.</div>
    <div class="body" style="margin-top: 10px">As small as 17 million parameters, answering in under a millisecond.</div></div></div>
  <div class="center"><div class="words h2">
    <span id="s3w0" class="a">Faster.</span><span id="s3w1" class="a">Cheaper.</span><span id="s3w2" class="a"><span class="grad-green">Just as accurate.</span></span>
  </div></div>
  <div id="s3foot" class="foot a">banking77 test set: Claude Opus 5.5 92%, a 17M-parameter Ettin student model 91.95%.</div>
  <div class="center"><div id="s3but" class="h2 a" style="max-width: 1500px">But few teams know how<br>to fine-tune one.</div></div>

  <!-- ============ S4: reveal ============ -->
  <div class="center"><div id="s4intro" class="h2 gray a">Introducing</div></div>
  <div class="center" style="top: -40px">
    <div id="s4icon" class="icon a"><svg width="112" height="112" viewBox="0 0 100 100"><circle cx="50" cy="50" r="27" stroke="#fff" stroke-width="10" fill="none"/><path d="M25 75 L75 25" stroke="#fff" stroke-width="10" stroke-linecap="round"/></svg></div>
    <div id="s4word" class="wordmark a" style="margin-top: 36px"><span class="grad" id="s4grad">SystemNone</span></div>
  </div>
  <div class="center" style="top: 330px"><div>
    <div id="s4t1" class="h3 a">Zero-click fine-tuning, performance and cost optimization.</div>
    <div id="s4t2" class="h3 gray a">Behind the API you already call.</div>
  </div></div>

  <!-- ============ S5: how it works ============ -->
  <div id="s5eyebrow" class="a eyebrow" style="left: 0; width: 1920px; top: 96px; text-align: center; color: var(--cyan)">How it works</div>
  <div class="center" style="top: -330px"><div id="s5t1" class="h3 a">Day one: your System One model answers.</div></div>
  <div class="center" style="top: -330px"><div id="s5t2" class="h3 a">SystemNone logs the inputs.</div></div>
  <div class="center" style="top: -330px"><div id="s5t3" class="h3 a">At 10,000 requests, a student model is trained for you.</div></div>
  <div class="center" style="top: -330px"><div id="s5t4" class="h3 a">The gateway loads it. Your app changes nothing.</div></div>

  <svg id="s5lines" class="a" width="1920" height="1080" style="left:0;top:0">
    <g stroke="#3a3a3c" stroke-width="2" fill="none" stroke-linecap="round">
      <path d="M440 447 H660 M440 483 H660"/>
      <path d="M1220 447 H1480 M1220 483 H1480"/>
    </g>
    <g stroke="#9a9aa0" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <path d="M648 441 L656 447 L648 453"/><path d="M452 477 L444 483 L452 489"/>
      <path d="M1468 441 L1476 447 L1468 453"/><path d="M1232 477 L1224 483 L1232 489"/>
    </g>
  </svg>
  <svg id="s5storeline" class="a" width="1920" height="1080" style="left:0;top:0"><path d="M940 598 V688" stroke="#3a3a3c" stroke-width="2" stroke-dasharray="2 8" stroke-linecap="round"/></svg>
  <svg id="s5workline" class="a" width="1920" height="1080" style="left:0;top:0"><path d="M1110 752 H1196" stroke="#3a3a3c" stroke-width="2" stroke-dasharray="2 8" stroke-linecap="round"/></svg>

  <div id="s5app" class="node a" style="left: 140px; top: 380px; width: 300px; height: 170px">
    <div class="t">Your app</div><div class="s mono" style="font-size: 19px">POST /v1/systemone</div>
  </div>
  <div id="s5gw" class="node a" style="left: 660px; top: 335px; width: 560px; height: 262px; justify-content: flex-start; padding-top: 30px; align-items: center">
    <div class="t" style="font-size: 38px">SystemNone gateway</div><div class="s">Rust</div>
  </div>
  <div id="s5slot" class="slot a" style="left: 770px; top: 504px">no student model yet</div>
  <div id="s5s1" class="node a" style="left: 1480px; top: 380px; width: 300px; height: 170px">
    <div class="t">System One</div><div class="s">model or LLM</div>
  </div>
  <div id="s5store" class="node a" style="left: 770px; top: 690px; width: 340px; height: 124px; padding: 0 28px">
    <div class="t" style="font-size: 30px">Store</div>
    <div class="s tnum"><span id="s5cnt">0</span> / 10,000 inputs</div>
  </div>
  <div id="s5ready" class="a" style="left: 1098px; top: 648px"><span class="pill" style="background: rgba(48,209,88,.18); color: var(--green)">Ready to train</span></div>
  <div id="s5work" class="node a" style="left: 1200px; top: 690px; width: 650px; height: 124px; padding: 0 26px">
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px">
      <div id="s5st0" class="step"><i class="fill"></i><span>Judge</span></div>
      <div id="s5st1" class="step"><i class="fill"></i><span>Fine-tune</span></div>
      <div id="s5st2" class="step"><i class="fill"></i><span>Calibrate</span></div>
      <div id="s5st3" class="step"><i class="fill"></i><span>ONNX</span></div>
    </div>
  </div>
  <div id="s5worklab" class="a" style="left: 1200px; top: 832px; width: 650px; text-align: center; font-size: 22px; color: var(--gray); font-weight: 550">Worker · Python</div>
  <div id="s5stu" class="stu a" style="left: 0; top: 0"><svg width="26" height="26" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>Student model</div>
  <div id="s5share" class="a" style="left: 140px; top: 600px; width: 400px">
    <div style="font-size: 22px; color: var(--gray); font-weight: 550">Answered by the student model</div>
    <div class="tnum" style="font-size: 84px; font-weight: 700; letter-spacing: -0.04em; color: var(--green)"><span id="s5pct">0</span>%</div>
  </div>
  <canvas id="s5cv" width="1920" height="1080" style="left:0;top:0"></canvas>

  <!-- ============ S6: calibration ============ -->
  <div class="center" style="top: -300px"><div id="s6title" class="h2 a">Confidence you can trust.</div></div>
  <div class="center" style="top: 60px"><div id="s6b" class="a">
    <div class="h3" style="margin-bottom: 50px">You choose the accuracy. <span class="gray">SystemNone picks the threshold.</span></div>
    <div id="s6target" class="a" style="margin-bottom: 26px"><span class="pill" style="font-size: 26px; padding: 8px 22px; background: rgba(10,132,255,.2); color: #5eb0ff">Target: 95% accuracy</span></div>
    <div class="bar"><i id="s6fill" style="width: 0"></i></div>
    <div id="s6labs" class="a" style="display: flex; justify-content: space-between; width: 1300px; margin-top: 20px; font-size: 28px; font-weight: 600">
      <span style="color: var(--green)">The student model answers 89%</span><span class="gray">System One answers the rest</span>
    </div>
  </div></div>
  <div id="s6foot2" class="foot a">banking77 test set, Ettin 17M student model, 95% accuracy target.</div>

  <!-- ============ S7: numbers ============ -->
  <div class="center" style="top: -60px"><div id="s7n1" class="mega tnum a"><span class="grad-green" id="s7v1">1×</span></div></div>
  <div class="center" style="top: 150px"><div id="s7l1" class="h2 a">faster</div></div>
  <div class="center" style="top: 250px"><div id="s7s1" class="body a">0.96 ms instead of 2.4 s</div></div>

  <div class="center" style="top: -60px"><div id="s7n2" class="mega a"><span class="grad-green">≈ $0</span></div></div>
  <div class="center" style="top: 150px"><div id="s7l2" class="h2 a">per million messages</div></div>
  <div class="center" style="top: 250px"><div id="s7s2" class="body a">instead of $8,120</div></div>

  <div class="center" style="top: -60px"><div id="s7n3" class="mega tnum a"><span class="grad-green" id="s7v3">0%</span></div></div>
  <div class="center" style="top: 150px"><div id="s7l3" class="h2 a">accurate</div></div>
  <div class="center" style="top: 250px"><div id="s7s3" class="body a">A frontier LLM scored 92%.</div></div>

  <div class="center" style="top: -60px"><div>
    <div id="s7t1" class="h2 a">Trained in 10 minutes.</div>
    <div id="s7t2" class="h2 a"><span class="grad">On a CPU.</span></div>
  </div></div>
  <div id="s7foot" class="foot a">banking77, 77 labels. Ettin 17M trained by SystemNone on banking77's human labels, scored on 3,080 test messages, ONNX on CPU. Claude Opus 5.5 scored on 200 messages (±2 points).</div>

  <!-- ============ why it's fast: what each request carries ============ -->
  <div class="center"><div id="rqq" class="h2 a">Why is the student model so fast?</div></div>
  <div class="center" style="top: -330px"><div id="rqt1" class="h3 a">An LLM or a System One model gets the prompt and the labels<br><span class="gray">with every request, on top of the input.</span></div></div>
  <div class="center" style="top: -330px"><div id="rqt2" class="h3 a">A student model learned them in training.<br><span class="gray">It only needs the input.</span></div></div>
  <div id="rqrows"></div>
  <div class="center" style="top: 350px"><div id="rqkick" class="h3 a">Less to read, every time: <span class="grad-green">one reason it's faster.</span></div></div>

  <!-- ============ inside the models ============ -->
  <div id="imeye" class="a eyebrow" style="left: 0; width: 1920px; top: 96px; text-align: center; color: var(--gray)">Inside the models</div>
  <div class="center" style="top: -330px"><div id="imt1" class="h3 a">An LLM turns each token into a vector<br><span class="gray">and runs the vectors through its layers.</span></div></div>
  <div class="center" style="top: -330px"><div id="imt2" class="h3 a">It writes its answer one token at a time,<br><span class="gray">running every layer again for each one.</span></div></div>
  <div class="center" style="top: -330px"><div id="imt3" class="h3 a">A System One model skips the loop.<br><span class="gray">One pass through its layers scores the labels.</span></div></div>
  <div class="center" style="top: -330px"><div id="imt4" class="h3 a">A student model is simpler still:<br><span class="gray">fewer layers, smaller vectors, only the input.</span></div></div>
  <canvas id="imcv" width="1920" height="1080" style="left:0;top:0"></canvas>
  <div class="center" style="top: 390px"><div id="imkick" class="h3 a">One small pass, <span class="grad-green">under a millisecond.</span></div></div>
  <div id="imfoot1" class="foot a">GPT-3 175B's published sizes (Brown et al., 2020); frontier models don't publish theirs. The tokens and scores are illustrative.</div>
  <div id="imfoot2" class="foot a">As Decider-2B does: it reads every option's probability from one forward pass, without generating a token.</div>
  <div id="imfoot3" class="foot a">Ettin 17M's published sizes: 7 layers, 256 numbers per token. 0.96 ms per answer on banking77, ONNX on a CPU.</div>

  <!-- ============ S8: escalation ============ -->
  <div class="center"><div id="s8q" class="h2 a">And when System One isn't sure?</div></div>
  <div id="s8eye" class="a eyebrow" style="left: 0; width: 1920px; top: 118px; text-align: center; color: var(--orange)">New: escalation</div>
  <div class="center" style="top: -300px"><div id="s8title" class="h2 a">It asks a stronger model.</div></div>
  <svg id="s8lines" class="a" width="1920" height="1080" style="left:0;top:0">
    <g stroke="#3a3a3c" stroke-width="2" fill="none" stroke-linecap="round"><path d="M70 540 H245 M600 540 H785 M1140 540 H1325"/></g>
    <g stroke="#9a9aa0" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M233 532 L243 540 L233 548"/><path d="M773 532 L783 540 L773 548"/><path d="M1313 532 L1323 540 L1313 548"/></g>
  </svg>
  <div id="s8al1" class="arrowlab a" style="left: 592px; top: 490px">unsure</div>
  <div id="s8al2" class="arrowlab a" style="left: 1132px; top: 490px">still unsure</div>
  <div id="s8n0" class="tier a" style="left: 250px; top: 450px"><div class="t" style="color: var(--green)">Student model</div><div class="s">fastest</div></div>
  <div id="s8n1" class="tier a" style="left: 790px; top: 450px"><div class="t" style="color: var(--cyan)">System One</div><div class="s">model</div></div>
  <div id="s8n2" class="tier a" style="left: 1330px; top: 450px"><div class="t" style="color: var(--orange)">LLM</div><div class="s">strongest</div></div>
  <div class="center" style="top: 290px"><div id="s8cap" class="h3 gray a">Classifier, System One, LLM: <span style="color: var(--text)">a three-step cascade.</span></div></div>
  <canvas id="s8cv" width="1920" height="1080" style="left:0;top:0"></canvas>

  <!-- ============ human labels ============ -->
  <div class="center" style="top: -300px"><div id="hltitle" class="h2 a">It learns from people, too.</div></div>
  <div class="center" style="top: -178px"><div id="hlsub" class="body a">You can upload datasets with human labels, which are superior to those from LLM judges.</div></div>
  <div class="center" style="top: 90px"><div class="hrows">
    <div id="hlr0" class="hrow a"><div class="hl"><span>The teacher: an LLM judge</span><span id="hlv0" class="tnum" style="color: var(--orange)">0%</span></div>
      <div class="bar"><i id="hlf0" style="width: 0; background: linear-gradient(90deg, #b36200, #ff9f0a)"></i></div></div>
    <div id="hlr1" class="hrow a"><div class="hl"><span class="gray">Student model, taught by the judge</span><span id="hlv1" class="tnum gray">0%</span></div>
      <div class="bar"><i id="hlf1" style="width: 0; background: linear-gradient(90deg, #1d5e33, #2f8f4f)"></i></div></div>
    <div id="hlr2" class="hrow a"><div class="hl"><span>Student model, taught by people</span><span id="hlv2" class="tnum" style="color: var(--green)">0%</span></div>
      <div class="bar" id="hlbar2"><i id="hlf2" style="width: 0"></i></div></div>
    <i id="hlmark" class="a" style="position: absolute; left: 1059px; top: 38px; height: 300px; width: 0; border-left: 3px dashed rgba(255,159,10,.75)"></i>
  </div></div>
  <div class="center" style="top: 350px"><div id="hlkick" class="h3 a">The student model can <span class="grad-green">surpass its teacher.</span></div></div>
  <div id="hlfoot" class="foot a">banking77, Ettin 17M student model. With Kimi K3 as the judge (81.6%), it scored 79.5% on the judge's labels and 91.5% on human labels.</div>

  <!-- ============ S9: careful by default ============ -->
  <div class="center" style="top: -250px"><div id="s9title" class="h2 a">Careful with your data.</div></div>
  <div class="center" style="top: 110px"><div class="cols">
    <div id="s9c1" class="col a">
      <div class="ic" style="background: rgba(255,159,10,.18)"><svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#ffb340" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M10 13h6M10 17h4"/></svg></div>
      <div class="t">License-aware.</div><div class="s">Judges whose terms forbid distillation are refused.</div>
    </div>
    <div id="s9c2" class="col a">
      <div class="ic" style="background: rgba(48,209,88,.18)"><svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#30d158" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l8 3v6c0 4.5-3.4 8.2-8 9-4.6-.8-8-4.5-8-9V6z"/><circle cx="12" cy="10" r="2.6"/><path d="M7.8 16.8c.9-1.9 2.4-2.9 4.2-2.9s3.3 1 4.2 2.9"/></svg></div>
      <div class="t">Private training.</div><div class="s">Training runs on your servers. With a self-hosted judge, no closed-model provider sees personal data.</div>
    </div>
  </div></div>

  <!-- ============ S10: product ============ -->
  <div class="center" style="top: -432px"><div id="s10cap" class="h3 a">Every prompt, and who answered it.</div></div>
  <div class="center" style="top: 70px"><div id="s10win" class="win a">
    <div class="bar0"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i></div>
    <div class="dash">
      <header><b>SystemNone</b><a class="on">Usage</a><a>Admin</a><span style="margin-left:auto;color:#a09f98;font-size:14px">Updated just now</span></header>
      <main>
        <div class="tiles">
          <div class="tile"><div class="k">Requests</div><div class="v" id="d0">1,200,063</div></div>
          <div class="tile"><div class="k">Today</div><div class="v" id="d1">46,491</div></div>
          <div class="tile"><div class="k">Answered by student models</div><div class="v" style="color:#4cc49a" id="d2">91%</div></div>
          <div class="tile"><div class="k">Student models live</div><div class="v">3</div></div>
        </div>
        <h2>Most used prompts</h2>
        <div class="tbl" id="dtbl"></div>
      </main>
    </div>
  </div></div>
  <div class="center"><div id="s10wall" class="wall a"></div></div>

  <!-- ============ S11: close ============ -->
  <div class="center"><div id="s11term" class="term a">
    <div class="bar0"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i></div>
    <pre id="s11pre"></pre>
  </div></div>
  <div class="center" style="top: -60px">
    <div id="s11icon" class="icon a"><svg width="112" height="112" viewBox="0 0 100 100"><circle cx="50" cy="50" r="27" stroke="#fff" stroke-width="10" fill="none"/><path d="M25 75 L75 25" stroke="#fff" stroke-width="10" stroke-linecap="round"/></svg></div>
    <div id="s11word" class="wordmark a" style="margin-top: 36px"><span class="grad" id="s11grad">SystemNone</span></div>
  </div>
  <div class="center" style="top: 300px"><div id="s11tag" class="h3 gray a" style="font-weight: 550">Faster answers. <span style="color: var(--text)">Same API.</span></div></div>
  <div class="center" style="top: 392px"><a id="s11link" class="mono a" href="https://github.com/JaderDias/SystemNone" target="_blank" rel="noopener"
    style="font-size: 34px; font-weight: 500; color: var(--cyan); text-decoration: none; pointer-events: auto">github.com/JaderDias/SystemNone</a></div>

  <!-- ============ other languages: a bigger multilingual student ============ -->
  <div class="center"><div id="mlq" class="h2 a">Other languages?</div></div>
  <div class="center" style="top: -330px"><div id="mlt1" class="h3 a">A multilingual student must be bigger:<br><span class="gray">its vocabulary and its vectors hold every language.</span></div></div>
  <div class="center" style="top: -330px"><div id="mlt2" class="h3 a">It takes longer to train,<br><span class="gray">and it wants a GPU.</span></div></div>
  <canvas id="mlcv" width="1920" height="1080" style="left:0;top:0"></canvas>
  <div class="center" style="top: 418px"><div id="mlkick" class="h3 a">Still faster than System One, <span class="grad-green">and cheaper to run.</span></div></div>
  <div id="mlfoot" class="foot a">Published sizes: Ettin 17M, 50,368 tokens × 256 numbers; mmBERT-base, 256,000 tokens × 768 numbers, 307M parameters.</div>

  <!-- ============ other languages: a student per language ============ -->
  <div class="center"><div id="rtq" class="h2 a" style="max-width: 1500px">Or: a <span class="grad-green">tiny</span> student<br>for each language.</div></div>
  <div class="center" style="top: -330px"><div id="rtt1" class="h3 a">A language router reads each message first<br><span class="gray">and sends it to its language's own student.</span></div></div>
  <svg id="rtlines" class="a" width="1920" height="1080" style="left:0;top:0">
    <g stroke="#3a3a3c" stroke-width="2" fill="none" stroke-linecap="round">
      <path d="M560 340 C620 340 600 515 690 515 M560 515 H690 M560 690 C620 690 600 515 690 515"/>
      <path d="M1050 515 C1150 515 1150 335 1250 335 M1050 515 C1150 515 1150 485 1250 485 M1050 515 C1150 515 1150 635 1250 635"/>
    </g>
  </svg>
  <div id="rtm0" class="rtmsg a" style="left: 110px; top: 310px">I was charged twice.</div>
  <div id="rtm1" class="rtmsg a" style="left: 110px; top: 485px">Cobraram-me duas vezes.</div>
  <div id="rtm2" class="rtmsg a" style="left: 110px; top: 660px">मुझसे दो बार पैसे लिए गए।</div>
  <div id="rtrouter" class="node a" style="left: 690px; top: 430px; width: 360px; height: 170px; align-items: center">
    <div class="t">Language router</div><div class="s">which language?</div>
  </div>
  <div id="rts0" class="rtstu a" style="left: 1250px; top: 280px"><b>Student</b><span>English</span></div>
  <div id="rts1" class="rtstu a" style="left: 1250px; top: 430px"><b>Student</b><span>Português</span></div>
  <div id="rts2" class="rtstu a" style="left: 1250px; top: 580px"><b>Student</b><span>हिन्दी</span></div>
  <canvas id="rtcv" width="1920" height="1080" style="left:0;top:0"></canvas>
  <div class="center" style="top: 395px"><div id="rtkick" class="h3 a">Each student stays tiny: <span class="grad-green">fast on a CPU.</span></div></div>
  <div id="rtfoot" class="foot a">The router: a small trigram model in the gateway. A language gets its own student once the router is sure of enough of its requests (300 by default).</div>

  <!-- ============ other languages: the cache ============ -->
  <div id="caeye" class="a eyebrow" style="left: 0; width: 1920px; top: 118px; text-align: center; color: #5eb0ff">New: the cache</div>
  <div class="center" style="top: -300px"><div id="catitle" class="h2 a">Seen it before?</div></div>
  <div class="center" style="top: -190px"><div id="casub" class="h3 gray a">The cache answers with what System One or the LLM said last time.</div></div>
  <svg id="calines" class="a" width="1920" height="1080" style="left:0;top:0">
    <g stroke="#3a3a3c" stroke-width="2" fill="none" stroke-linecap="round"><path d="M20 540 H1370"/></g>
    <g stroke="#9a9aa0" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M278 532 L288 540 L278 548"/><path d="M818 532 L828 540 L818 548"/><path d="M1358 532 L1368 540 L1358 548"/></g>
  </svg>
  <div id="can0" class="tier a" style="left: 290px; top: 465px; width: 300px; height: 150px"><div class="t" style="color: var(--green)">Student model</div><div class="s">fastest</div></div>
  <div id="can1" class="tier a" style="left: 830px; top: 465px; width: 300px; height: 150px"><div class="t" style="color: var(--cyan)">System One</div><div class="s">model</div></div>
  <div id="can2" class="tier a" style="left: 1370px; top: 465px; width: 300px; height: 150px"><div class="t" style="color: var(--orange)">LLM</div><div class="s">strongest</div></div>
  <div id="cacard" class="tier a" style="left: 0; top: 480px; width: 200px; height: 120px; border-color: rgba(94,176,255,.6)"><div class="t" style="color: #5eb0ff; font-size: 34px">Cache</div><div class="s" style="font-size: 22px">seen before?</div></div>
  <canvas id="cacv" width="1920" height="1080" style="left:0;top:0"></canvas>
  <div class="center" style="top: 170px"><div id="capos0" class="body a">Before the student model</div></div>
  <div class="center" style="top: 170px"><div id="capos1" class="body a">Between the student model and System One</div></div>
  <div class="center" style="top: 170px"><div id="capos2" class="body a">Between System One and the LLM</div></div>
  <div class="center" style="top: 330px"><div id="cakick" class="h3 a">Anywhere you put it, <span class="grad-green">it saves every step after it.</span></div></div>

  <div id="fade" style="left:0;top:0;width:1920px;height:1080px;background:#000;opacity:0"></div>
</div>
`);
