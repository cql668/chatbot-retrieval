## Retrieval-Based Conversational Model in Tensorflow (Ubuntu Dialog Corpus)

#### [Please read the blog post for this code](http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow)

#### Overview

The code here implements the Dual LSTM Encoder model from [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](http://arxiv.org/abs/1506.08909).

#### Setup

This code uses Python 3 and Tensorflow >= 0.9. Clone the repository and install all required packages:

```
pip install -U pip
pip install numpy scikit-learn pandas jupyter
```

#### Get the Data


Download the train/dev/test data [here](https://drive.google.com/open?id=0B_bZck-ksdkpVEtVc1R6Y01HMWM) and extract the acrhive into `./data`.


#### Training

```
python udc_train.py
```


#### Evaluation

```
python udc_test.py --model_dir=...
```


#### Evaluation

```
python udc_predict.py --model_dir=...
```


      这次测试的操作系统依然是Ubuntu14.04(64位)。

      开源项目链接：https://github.com/dennybritz/chatbot-retrieval/

      它实现一个检索式的机器人。采用检索式架构，有预定好的语料答复库。检索式模型的输入是上下文潜在的答复。模型输出对这些答复的打分，选择最高分的答案作为回复。

      下面进入正题。

      1.环境配置

      首先此项目需要的基本条件是使用Python3(我用的是Python3.4)，tensorflow版本为0.11.0。关于Python这里不多说，网上很多修改Python默认值的文章。后续内容我都将采用python3或者pip3指令，在Python3下进行操作。tensorflow在我测试时，过低版本或者新版本都会出现一些问题，所以建议和我采用一样的版本（因为我的电脑是AMD的显卡，所以我没有选择GPU版本的tensorflow，有条件的可以选择）。如果不是可以采用以下命令修改：

sudo pip3 uninstall tensorflow

sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.11.0-cp34-cp34m-linux_x86_64.whl

      好了，最基本的条件准备好了。

      下面我们将安装一系列的依赖包，因为里面依赖关系比较复杂，所以需要注意一下先后顺序。我们在命令行中依次输入以下指令：

sudo apt-get upgrade python3-pip

sudo pip3 install numpy scipy scikit-learn pandas pillow jupyter

sudo pip3 install backports.weakref==1.0rc1

sudo apt-get build-dep Python-imaging

sudo pip3 install tflearn

      到这里，我们的环境基本配置好了，但是因为我是在配置过程中一个一个去解决的这些问题，中间碰到的问题也比较多，这是重新整理过的，每个人的环境也都有所差异，我也不能确保这样就完全正确。一般这里面碰到问题无非就是两种，一是缺包，二是tensorflow版本的问题，往这两方面考虑就可以解决。

      最后，检查一下这些工具是否都安装，我们开始导入数据。

      首先执行git clone https://github.com/dennybritz/chatbot-retrieval/

      然后到此链接https://drive.google.com/file/d/0B_bZck-ksdkpVEtVc1R6Y01HMWM/view（需要fanqiang）去下载数据，页面如果显示“糟糕，出现预览问题，正在重新加载”，不要管它，点击下载即可。将下载到的数据解压到刚才clone的文件夹chatbot-retrieval的data中，如图所示：



      2.训练与测试

      在chatbot-retrieval文件夹中打开终端，或者cd到该文件夹下，执行以下指令：

python3 udc_train.py

      即可开始训练，我采用CPU训练了三个多小时，这个视个人情况而定，可以按Ctrl+Z提前结束。在GPU上训练2万次大约一个多小时。正常训练过程中如图所示：


      训练完后仍然在刚才的路径下可以执行以下命令对模型进行评估（可以跳过）：

python3 udc_test.py –model_dir=./runs/1504264339/

其中后面的数字名称的文件夹名因人而异，不同的训练批次名称也不一样。这个名称在训练的那张截图里也可以发现。

      最后，进入我们的测试环节。

      找到chatbot-retrieval文件夹下的udc_predict.py文件，将30行INPUT_CONTEXT =后的内容改成自己想要问的内容，将31行POTENTIAL_RESPONSES = []中的内容替换成机器人的候选答案。因为这个项目没有实现一问一答的模式，每次只是给不同的答案进行打分，分数最高的那个是它的期望回答，所以下面我们都将以其回答中打分最高的回答作为标准判断正确率。仍然在chatbot-retrieval文件夹路径下执行python3udc_predict.py --model_dir=./runs/1504221361/指令进行测试。

      下面是测试情况：

中文闲聊型：
      （1）INPUT_CONTEXT = "你好"

      POTENTIAL_RESPONSES = ["你好", "早上好","中午好","晚上好","好啊","好久不见","很高兴认识你","初次见面请多多指教","我不好","你是谁",]

      测试效果如图，我们将每个结果的打分筛选出来。


Context: 你好

你好: 0.501835

早上好: 0.501835

中午好: 0.501835

晚上好: 0.501835

好啊: 0.501835

好久不见: 0.501835

很高兴认识你: 0.501835

初次见面请多多指教: 0.501835

我不好: 0.501835

你是谁: 0.501835

      可以看到所有回答的打分都是一样的，这其实是因为语料库采用了Ubuntu对话数据集，无法处理中文。我们再测一组中文进行验证。

 

      （2）INPUT_CONTEXT = "明天上午啥课？"

      POTENTIAL_RESPONSES = ["明天上午没课", "计算机图形学和形式与政策","明天上午有课吗","还没开学好不好","包子和稀饭","超市没开门","明天下雨","那一年你正年轻","时间是让人猝不及防的东西","瞎扯",]

测试结果：

Context: 明天上午啥课？

明天上午没课: 0.501835

计算机图形学和形式与政策: 0.501835

明天上午有课吗: 0.501835

还没开学好不好: 0.501835

包子和稀饭: 0.501835

超市没开门: 0.501835

明天下雨: 0.501835

那一年你正年轻: 0.501835

时间是让人猝不及防的东西: 0.501835

瞎扯: 0.501835

 

      这验证了我们前面的结论，该机器人无法对中文进行判断。接下来，我们测试英文话题。

英文闲聊型
      （1）INPUT_CONTEXT = "hello"

       POTENTIAL_RESPONSES =["hi", "who are you","how are you","whereare you come from","how old are you"]

测试结果：

Context: hello

hi: 0.515187

who are you: 0.452673

how are you: 0.485824

where are you come from:0.462595

how old are you: 0.505551

      最高分结果为hi

      （2）INPUT_CONTEXT = "Where are you going?"

       POTENTIAL_RESPONSES =["The weather is nice today", "I love you","Are yousure?","Go home","My name is Watson."]

测试结果：

Context: Where are yougoing?

The weather is nice today:0.542136

I love you: 0.570628

Are you sure?: 0.564842

Go home: 0.528381

My name is Watson.:0.567629

      最高分结果为I love you

      （3）INPUT_CONTEXT = "What's your name?"

      POTENTIAL_RESPONSES =["No problem", "What a beautiful girlfriend youhave!","Look over there","My favorite basketball player isMichael Jordan","My name is Watson."]

测试结果：

Context: What's your name?

No problem: 0.433695

What a beautifulgirlfriend you have!: 0.53788

Look over there: 0.502499

My favorite basketballplayer is Michael Jordan: 0.557139

My name is Watson.:0.568081

      最高分结果为My name is Watson.

      （4）INPUT_CONTEXT = "Where are you come from?"

      POTENTIAL_RESPONSES =["China", "You're welcome","I am aboy","I've come up with a good idea","Don't go there"]

测试结果：

Context: Where are youcome from?

China: 0.500012

You're welcome: 0.505979

I am a boy: 0.498553

I've come up with a goodidea: 0.540985

Don't go there: 0.582593

      最高分结果为Don't go there

      （5）INPUT_CONTEXT = "How much is this dress?"

      POTENTIAL_RESPONSES =["15 dollars", "hello","I like dogs","It'sreally hot today","Don't worry"]

测试结果为：

Context: How much is thisdress?

15 dollars: 0.553268

hello: 0.500417

I like dogs: 0.493128

It's really hot today:0.665182

Don't worry: 0.531877

      最高分答案为It's really hot today

 

      由以上五组测试结果可以看到，回答正确的为（1）（3），正确率为40%。

英文任务型
     （1）INPUT_CONTEXT = "Book me a ticket from Wuhan to Nanjingtomorrow."

      POTENTIAL_RESPONSES =["100 dollars", "There will be two flight tomorrow. Which flightwill you take?","You are handsome"]

测试结果：

Context: Book me a ticketfrom Wuhan to Nanjing tomorrow.

100 dollars: 0.460377

There will be two flighttomorrow. Which flight will you take?: 0.50801

You are handsome: 0.524509

      最高分答案为You are handsome

      （2）INPUT_CONTEXT = "Help me find out what the weather willbe like tomorrow"

      POTENTIAL_RESPONSES =["It's going to be sunny tomorrow", "I'm fine.Thankyou.","That's a cool car"]

测试结果：

Context: Help me find outwhat the weather will be like tomorrow

It's going to be sunnytomorrow: 0.480107

I'm fine.Thank you.:0.412907

That's a cool car:0.481291

      最高分答案为That's a cool car

（3）INPUT_CONTEXT = "Get me a KFC take out"

POTENTIAL_RESPONSES =["All right. I've already placed your order", "I need a bottleof mineral water","Deal"]

      测试结果：

Context: Get me a KFC takeout

All right. I've alreadyplaced your order: 0.468371

I need a bottle of mineralwater: 0.535811

Deal: 0.468816

      最高分答案为I need a bottle of mineral water

      （4）INPUT_CONTEXT = "Check out the nearest hotel"

      POTENTIAL_RESPONSES =["He looks so happy", "The hotel is 500 dollars anight","The nearest hotel is 500 meters away from you. Here's yournavigation route"]

测试结果：

Context: Check out thenearest hotel

He looks so happy:0.541198

The hotel is 500 dollars anight: 0.458123

The nearest hotel is 500meters away from you. Here's your navigation route: 0.433537

      最高分答案为He looks so happy

      （5）INPUT_CONTEXT = "Call a taxi for me"

      POTENTIAL_RESPONSES =["I've got a taxi for you. It will arrive in 5 minutes", "Thecar is worth 20 thousand dollars","Is that your car?"]

测试结果：

Context: Call a taxi forme

I've got a taxi for you.It will arrive in 5 minutes: 0.523356

The car is worth 20thousand dollars: 0.53095

Is that your car?:0.444959

      最高分答案为The car is worth 20 thousand dollars



      可以看到，正确率为0。

英文知识型
      （1）INPUT_CONTEXT = "Is Shakespeare a male or afemale?"

      POTENTIAL_RESPONSES =["male", "female"]

测试结果：

Context: Is Shakespeare amale or a female?

male: 0.535856

female: 0.498352

     最高分答案为male

     （2）INPUT_CONTEXT = "Who wrote the lady of the camellias？"

      POTENTIAL_RESPONSES =["Alexandre Dumas.fils", "Shakespeare","leotolstoy"]

测试结果：

Context: Who wrote thelady of the camellias？

Alexandre Dumas.fils:0.587284

Shakespeare: 0.539719

leo tolstoy: 0.477065

      最高分答案为Alexandre Dumas.fils

      （3）INPUT_CONTEXT = "When British constitutional monarchywas established？"

      POTENTIAL_RESPONSES =["1688", "1840","1949"]

测试结果：

When Britishconstitutional monarchy was established？

1688: 0.517922

1840: 0.517922

1949: 0.517922

      得分都一样，本题算错

      （4）INPUT_CONTEXT = "What country does Shanghai belong to？"

     POTENTIAL_RESPONSES =["China", "USA","Spain"]

测试结果：

Context: What country doesShanghai belong to？

China: 0.503242

USA: 0.503242

Spain: 0.503242

      得分都一样，本题算错

      （5）INPUT_CONTEXT = "What's Michael Jordan's father's firstname？"

      POTENTIAL_RESPONSES =["Jordan", "Li","Michael"]

Context: What's MichaelJordan's father's first name？

Jordan: 0.5339

Li: 0.5339

Michael: 0.5339

      得分都一样，本题算错

 

      由上可知，（1）（2）为正确的，正确率为40%。

 

      综上，chatbot对于闲聊和知识型问题准确率能做到40%，但是对于任务型问题存在很大的不足。
--------------------- 
作者：MrLittleDog 
来源：CSDN 
原文：https://blog.csdn.net/hfutdog/article/details/78155676 
版权声明：本文为博主原创文章，转载请附上博文链接！
