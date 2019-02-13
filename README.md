# ML-DailyWork
机器学习大作业，文本分类<br>
（1）机器学习期末作业，要求：取前60%作为训练集、最后的40%作为测试集；<br>
（2）采用KNN算法，测试集准确率为73%；<br>
（3）采用SVM算法，测试集准确率为77%；<br>
（4）对数据进行归一化，机器学习算法均在80%以下；<br>
（5）之后采用深度学习算法优化分类，采用了BN与Dropout；<br>
    首先我采用基本的全连接网络对数据进行分类，全连接设置了N次，最好的为128-128-128；最后做的识别率有81.7%；<br>
    之后我采用CNN进行分类，发现CNN只在81%左右；<br>
    后来我采用LSTM,GRU网络进行分类，大约能做到82%准确率；<br>
    然后采用CLDNN网络结构，做到了82.3%；<br>
    最后我将归一化数据与未归一化数据进行整合，最后做到82.7%。<br>
欢迎各位大佬提宝贵建议。<br>
