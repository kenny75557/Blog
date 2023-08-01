---
title: "Blog的第一篇，從2022年實習寫的論文開始好了"
description: "開始即結束？"
date: 2023-06-27T17:23:48+08:00
draft: False
# cover:
#     image: "post/first-post/images/inria_BOD.jpg"
#     alt: ""
#     caption: "INRIA Bordeaux"
categories: ["NLP"] 
tags: ["ACL2023","Paper","Intern"]
---
![INRIA Bordeaux](/post/first-post/inria_BOD.jpg "INRIA Bordeaux")

### 前言: Remote的海外實習之旅🇫🇷

21年11月的某天，這天是個星期二，我一如往常在 Lab 刷 arxiv 找論文，尋求一絲下週 Meeting 不會被汎老師釘在牆上的希望；隨著我打算放棄稍作休息時，看到了老師在社團分享的實習資訊。
![Intern Info.](/post/first-post/InternProject.png "Intern Info.")

嗯...基於好奇心提問的問題生成看起來是教育相關應用，NLP 的部分想要引入 GPT-3 來做題目生成及 Evaluation 吧？

再看一下單位：
Inria Bordeaux 跟 MSR Montreal 的合作，Microsoft Research 很難不知道，不過 Inria 就需要查一下了。

Inria, Institut National de Recherche en Informatique et en Automatique（法國國家信息與自動化研究所），順便發現Scikit Learn 是他家開發的。
雖然我的碩論方向不是做 QG 也不是 Prompt Engineering，但實習可以去法國半年，本人也不急著畢業找工作什麼的，於是種種誘因下我準備了履歷，麻煩老師幫我潤一下 Cover letter 之後就發射過去了。
那過程就是三場英文面試，ㄧ場講了做過的 NLP 專案另外兩場聊天居多，沒想到順利的上了XD

題外話，雖然專案是 Inria 那邊 Host，但後來我幾乎都跟 Montreal 那邊的 Team 做 LLM 相關的內容。 Mentor 也覺得我的情況做 NLP 比起去系統端更合適就邀我過去（畢竟學校這邊在寫碩論），然後提出了組內會議的時候可以講中文，原因我不好說...
![阿鬼 你還是說中文吧](/post/first-post/ghost.jpeg "阿鬼 你還是說中文吧")


如標題所說後來因疫情關係採遠端實習，所以變成一邊做碩論，口試時間也不用延後，外加他們會招待我一趟7天的參訪；
打聽了下，他們能給的補助也無法負擔在那邊的生活費，所以有送機票住宿還能準時畢業算是一個因禍得福的 Solution 吧。
至於我後來決定把波爾多一週參訪變成一個月的法國之旅又是另一個故事了。

廢話講完了進下一 Part 論文講解～

### Selecting Better Samples from Pre-trained LLMs: A Case Study on Question Generation 📖

##### **Introduction:** 

應用 GPT-3 生成引導學童提問系統所需的問題(Question Generation)時，當要求 LLM 生成多個問題，後面很容易回相同的問題或是太 General 的 Question (ex:結局是歡樂的嗎?)
最終目的是全自動化出題，除了優化 Prompt 及調 Temperature 參數外，還可以設計方法在這些 LLM outputs 中 Sample 出好的答案。

註：要提升 LM 的生成多樣性(Generation Divesity)常見的方法就是 Sampling ~~，關於 LM Sampling，我們之後會專門做一集視頻跟大家講解。~~


先前沒有穩定能從 LLM 的所有 Output 中 sample 最佳的方法，於是我們的方法就建立在以下背景：
1.  黑盒子模型 (non-modifiable 的 QG Model)
2.  缺乏人工標注的資料
    
而上述設定其實就是在現實中部署 LLM 會遇到的限制，只是每個 Industry 的需求問題不一樣。 
我們設計了自動評估及利用 Human Evaluation，來驗證所提出的方法所挑選出來的問題是否比 Greedy 策略（LLM 的第一筆生成結果，即最大機率）生成的結果更理想。





##### **Problem Setting**
跟一般 Question Generation 相同，Dataset 中的 Context-Answer Pairs (c,a) 都是字串，這個 Task 是要讓模型生成一個對應到答案 **a** 的問題 **q**，同時這個問題必須用 Context **c** 作為 supporting evidence。

不同的地方來了，在先前提到的設定中，對每一組Context-Answer我們會Sample **k**個問題，對照組則是 Greedy 生成的一個回答
(很抽象？其實就是調 Temperature 參數而已)。

所以到這邊每篇 Context 我們會有k個 sample questions 跟一個 Greedy 的 question；我們將k個 samples 的平均分數以及 Greedy 的分數作為兩個 Baseline (算分方式後面會介紹)
理想情況下實驗結果應該要優於這兩個 Baseline。
![Baseline](/post/first-post/Baseline.jpg "Baseline")
[Notion Define](https://arxiv.org/pdf/2209.11000.pdf)


##### **Dataset & Model**
資料集：

1. SQuAD : Sentence Level, ExtractiveQA.
2. Fairy-tail QA : Paragraph Level, Abstractive QA.
Zero-Shot setting，都只使用 TestingData。

模型：

GPT-3 text-davinci-003

[設定及Prompt可參考論文](https://arxiv.org/pdf/2209.11000.pdf)


##### **Evaluation Metrics**
我們使用兩個 metrics 來評估選出的**q'**
1. Reference-based evaluation
    * 沿用先前相關研究的指標，SQuAD 用的是 BLUE-4，FairyTale QA 用的是 ROUGE-L。
2. Human evaluation
    * 單純要標註者評估問題好壞肯定是高度主觀的，因此與 Inria 那邊共同設計了7個 dimension 跟一個 Overall 分數，我們稱為 Meta Question，讓標註者進行評分，每一組資料至少有三位 Anotator。

![MetaQuestion](/post/first-post/MetaQuestion.png "MetaQuestion")


##### **Method**
設計三種手法，兩個 Reference
1. **n-gram similarity:**
   
   * 用 n-gram 來衡量 Context 與生出 question 的關聯性，這個手法較為直接，Assume 與 Context 資訊重疊多的為品質好的問題，i.e., 這篇文章的標題應該是什麼，這類通用問句的 Similarity score 就會較低
2. **RoundTrip:**
   
   * 生成出的問題對應到的答案應該要與資料提供給生成的答案相同。也就是理想的情況下 QAQG model 的輸出應符合:

   ![RoundTrip](/post/first-post/RoundTrip.png "RoundTrip")
   ，此想法近似於 Cycle consistency 在圖片生成及翻譯上的應用。
   * 實作的話，將 GPT-3 生成的 Q 與 Context 重新輸入得到答案，與資料集 QA pair 中的 Answer 計算 F1Score (採用與資料集 Evaluation setup 相同的 BLUE-4, ROUGE)，最後 sample 能讓 GPT-3 生成出最高分答案的 Q。
   
3. **Prompt-based Score:**

   Inri 另一篇研究需要針對題目的品質，找 Annotater 做不同面向的評估，設計了七個維度的量表，包含文法正確性,與文章內容相不相關等等，於是直接拿了這個量表來反問 GPT-3，一個自評生成的好不好的概念。 
   分為兩步驟:

   ![humanEval](/post/first-post/humanEval.png "humanEval")

   * Step1. 將文本與 Sample 出的問題丟給 GPT-3，詢問上面提到的 Meta-Question，並要 Model 給出理由。
   * Step2. 讓 Model based on 問題, 文本, 以及生成的理由，回答一個分數（其實就是做跟 Human Annotater 相同的事）
這麼做的原因是我們觀察到如果沒有 Step1 要 model 給出理由，GPT-3 傾向 Low-entropy 的 Distribution，就是說同一個 Meta-Question 模型傾向於選擇同樣的分數，而不考慮 Context-Question pair。

至於為何加上的先詢問理由的步驟就會比較好，~~個人認為大部分的原因還是通靈~~，我覺得這邊的概念有點像 CoT(Chain of Thought)，只是我們當初沒有設計一套評分準則跟推理過程而是讓 GPT-3 自己 Inference。


最後，方法中使用的 OPS Score 及 APS Score 分別代表：

1. overall prompt-based score: 來自人工標注中其中一題，請給這個問題一個 Overall score。也同時問了 GPT-3，挑出最高的。 
2. averaged promptbased score: 七個 Meta Question 的平均分數最高的。

##### **實驗**


隨機從 GPT-3 Sample 的k個 Question 中使用設計的 Selection Method 來挑出 Question，使用前面提到的兩種 Evaluation評估。

此外也做了多種 method 的 Ensemble performance，將 Reference-based 及 Human Evaluation 的分數都 Normalize 到 0-1 之間以確保比較性。

###### **Reference-based evaluation:**
![Refbase_score](/post/first-post/Refbase_score.png "Refbase_score")

首先看到在 SQuAD 所有的 selction method 是高於 Average Score( 5個 Sample questions 平均)的，而各個 selction methods 也都優於 Greedy generation 的 Baseline。 要注意的是先前的比較對象都是經過 Label data 的 Fine-tuning，而我們用的是 Zero-shot GPT-3。

在 Fairy-Tale 只有 Upperbound 超過 BART 的 Performance，在 Abstractive QG 上，Selection method 還有蠻多空間。



###### **Human Evaluation:**
![HEresults](/post/first-post/HEresults.png "HEresults")

在 QG 以及其他 NLG task，LM 輸出的內容會是具有語言多樣性(linguistic diversity)但皆有相同語意(Semantic Equivalance)，因此使用 Single Reference 的 Evaluation 是不合適的(Ground Truth)。實驗中也發現，GT 的分數在大部分情況下都不是人為評估的最佳。
因此我們認為 Human Evalution 是必要且對提高 Performance 泛用到實際應用是有幫助的。


在實驗中，APS 以及 n-gram 的 Selection Method 在兩個資料集表現有較明顯的差異，在 SQuAD 資料集，n-gram 如同前面 Reference based 一樣表現得較好，加上 RoundTrip 的 Ensemble method 甚至可以更高，APS 則是在這些 meta-question 表現得最差。 

相反的 Fairy-Tale 資料集上 n-gram 表現的最差，APS 則是明顯的最優。
會有這樣的現象我們認為是 SQuAD 資料集的特性，也就是他的答案長度較短且都是從較短的 Context 中 Extract 出來的，因此參考上下文關聯性的 N-gram 在這種資料下就會是一個好的 Question Selection Method。

另一方面並不是所有需要問提生成的應用都是在這樣的出題模式(多段文章, abstractive)
，只能說若不在短文且答案在文章中的 Scenario，那麼 n-gram 不是一個通用且好的 Question Selection strategy。

**Discussion**
### 後記
回到現在，在前幾個月經歷了:

* 當個兵出來發現論文上 ACL (都跟朋友說我是：A real imposter, no syndrome.)

* 結訓出來求職（時機歹歹的科技業寒冬，過程有 frustrated 到；但不少機會沒能拿下確實是自己玩砸了Hehe）

* 訂完去多倫多的機票隔一小時拿到 Offer

後寫了這段發現半年過去，體感超快...；
其實試著寫部落格掛在嘴邊很久了，但總是因為懶或是覺得自己沒料（確實）而 Pending；
一個月前跟朋友討論後覺得以這個主題當第一篇，這樣去研討會的時候就可以日更，但拖延症大家都老熟了XDD

寫這篇的時間是23年7月4日的凌晨，明晚就要出發，還好趕上發布，那今天就先寫到這了咱們下篇再見。
